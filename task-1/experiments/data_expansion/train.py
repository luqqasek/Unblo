"""One bounded adaptation loop: matched old replay versus new PIPE constructions."""
import argparse
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch

from experiments.triforensic_wm.model import TriForensicWM
from experiments.triforensic_wm.data import Engine
from experiments.triforensic_wm.evaluate import summarize
from experiments.native_localizer.common import atomic_json, append_jsonl, rgb_tensor, seed_all, sha256, rng_state
from experiments.native_localizer.losses import segmentation
from experiments.native_localizer.evaluate import image_metrics
from wmloc.utils import downsample_mask


class PipePool:
    def __init__(self, root):
        folder = root/'data-expansion/encoded/pixelseal'
        assert all((folder/f'completed{p}.json').exists() for p in (0,1))
        self.rows = [json.loads(s) for p in (0,1) for s in (folder/f'train-part{p}.jsonl').read_text().splitlines()]
        self.data = []
        for r in self.rows:
            base = Path(r['path'])
            read = lambda p: np.asarray(Image.open(p)).copy()
            self.data.append((read(base/'clean.png'),read(base/'edited.png'),read(r['marked_path']),read(base/'edit_mask.png')>=128,r))
    def sample(self, rng):
        clean,edited,marked,mask,row = self.data[int(rng.integers(len(self.data)))]
        mode = int(rng.integers(8))
        if mode < 4:
            image = marked.copy(); image[mask] = (clean if mode < 2 else edited)[mask]
            support = ~mask; kind = 'restored_hole' if mode < 2 else 'inpainted_hole'
            assert np.array_equal(image[~mask],marked[~mask])
        elif mode < 6:
            image = edited.copy(); image[mask] = marked[mask]
            support = mask; kind = 'marked_object_remnant'
        else:
            image = clean if mode == 6 else marked
            support = np.full((512,512),mode==7,bool);kind = 'clean' if mode==6 else 'marked'
        return image,downsample_mask(support),row['message'],{'source_id':row['source_id'],'kind':kind}


def corrected(prob, local):
    result = prob.copy()
    if not (result>=.5).any():
        result.flat[np.argsort(local.ravel(),kind='stable')[-164:]] = 1
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=('old','pipe'),required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--wall-seconds',type=int,default=1450)
    a=p.parse_args();assert a.steps==200
    torch.set_num_threads(6);seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'data-expansion/runs'/a.arm;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'last.pt').exists(),'Preserve run; resumption must restore optimizer/RNG'
    start=time.monotonic();engine=Engine(a.root,'pixelseal');pipe=PipePool(a.root) if a.arm=='pipe' else None
    source=a.root/'triforensic-runs/pixelseal/unified-v1/resume.pt'
    state=torch.load(source,map_location='cpu',weights_only=False);assert state['step']==1561
    model=TriForensicWM('pixelseal',checkpointing=True).cuda();model.load_state_dict(state['ema']);del state
    model.carrier.requires_grad_(False);model.refiner.requires_grad_(False)
    groups={}
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            lr=2e-6 if name.startswith('base.encoder.') else 2e-5
            wd=.01 if parameter.ndim>1 else 0.
            groups.setdefault((lr,wd),[]).append(parameter)
    opt=torch.optim.AdamW([{'params':v,'lr':lr,'weight_decay':wd} for (lr,wd),v in groups.items()])
    split={'train':[engine.real[i]['image_name'] for i in engine.training],'validation':[engine.real[i]['image_name'] for i in engine.valid]}
    atomic_json(out/'split.json',split)
    assert split==json.loads((a.root/'triforensic-runs/pixelseal/unified-v1/split.json').read_text())
    config={'arm':a.arm,'steps':200,'batch':16,'microbatch':4,'real_views':4,'synthetic_views':12,'pipe_views':6 if pipe else 0,'source':str(source),'source_sha256':sha256(source),'carrier_fixed':True,'refiner_bypassed':True,'threshold':.5,'empty_fallback':'top164 unchanged carrier cells','post_embedding_augmentation':False,'loss':'per-image0.5BCE+0.5dualDice','encoder_lr':2e-6,'new_lr':2e-5}
    atomic_json(out/'config.json',config)
    def forward(x,b):
        with torch.no_grad():f,l,obs=model.local_forward(x)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model.global_forward(x,b,f,obs)[:,0]
        return z,l[:,0]
    @torch.inference_mode()
    def evaluate(step):
        model.eval();rows=[];raw=[];ps=[];ys=[];names=[];locals_=[]
        for begin in range(0,len(engine.valid),4):
            items=[engine.real[i] for i in engine.valid[begin:begin+4]]
            x=torch.stack([rgb_tensor(v['image']) for v in items]).cuda();b=torch.tensor(np.stack([v['message'] for v in items]),device='cuda').float()
            z,l=forward(x,b);prob=z.sigmoid().cpu().numpy();ll=l.cpu().numpy()
            for item,pp,loc in zip(items,prob,ll):
                pred=corrected(pp,loc);name=item['image_name'];target=item['mask128']
                raw.append(image_metrics(pp,target));rows.append(dict(image_metrics(pred,target),image_name=name))
                ps.append(pred);ys.append(target);names.append(name);locals_.append(loc)
        record={'step':step,'summary':summarize(rows),'raw_summary':summarize(raw),'per_image':rows,'protocol':'fixed reused57-image development fold; not full OOF'}
        if step==0:
            record['parent_raw_miou_difference']=record['raw_summary']['all']['miou']-.5817244623066825
            assert abs(record['parent_raw_miou_difference'])<.001,record['parent_raw_miou_difference']
        atomic_json(out/f'evaluation-{step}.json',record);np.savez_compressed(out/f'maps-{step}.npz',names=names,probabilities=ps,targets=ys,local_logits=locals_)
        print('PIPE_VALIDATION',a.arm,step,json.dumps(record['summary']),flush=True)
        model.train();model.carrier.eval();return record
    initial=evaluate(0);exposure=[]
    def batch(step):
        rng=np.random.default_rng(7421007+step);items=[];meta=[]
        for j in range(12):
            bucket=(step*12+j)%5;idx=int(rng.choice(engine.train.buckets[bucket]));x,y,b,_=engine.train.full(idx)
            items.append((x,y,b));meta.append({'source_id':engine.train.rows[idx]['source_id'],'kind':'old','index':idx})
        if pipe:
            nrng=np.random.default_rng(9921007+step)
            for j in range(6,12):
                x,y,b,m=pipe.sample(nrng);items[j]=(x,y,b);meta[j]=m
        for i in rng.choice(engine.training,4,replace=True):
            r=engine.real[int(i)];items.append((r['image'],r['mask128'],r['message']));meta.append({'real':r['image_name']})
        return (torch.stack([rgb_tensor(r[0]) for r in items]),torch.tensor(np.stack([r[1] for r in items])).float(),torch.tensor(np.stack([r[2] for r in items])).float(),meta)
    pool=ThreadPoolExecutor(max_workers=1);future=pool.submit(batch,0);training=time.monotonic();step=0;final=initial
    for update in range(a.steps):
        if time.monotonic()-start>a.wall_seconds:break
        x,y,b,meta=future.result()
        if update+1<a.steps:future=pool.submit(batch,update+1)
        opt.zero_grad(set_to_none=True);value=0.
        for j in range(0,16,4):
            z,_=forward(x[j:j+4].cuda(),b[j:j+4].cuda());loss=segmentation(z,y[j:j+4].cuda())/4;loss.backward();value+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step();step=update+1
        exposure.extend(meta)
        if step==1 or step%10==0:
            elapsed=time.monotonic()-training;row={'step':step,'loss':value,'gradient_norm':float(norm),'seconds_per_update':elapsed/step,'eta_seconds':(a.steps-step)*elapsed/step,'peak_gb':torch.cuda.max_memory_allocated()/1e9}
            append_jsonl(out/'train.jsonl',row);print('PIPE_TRAIN',a.arm,json.dumps(row),flush=True)
        if step in (50,200):final=evaluate(step)
    pool.shutdown(wait=True)
    if step!=200:final=evaluate(step)
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'step':step,'config':config},out/'last.pt')
    atomic_json(out/'exposure.json',{'views':len(exposure),'records':exposure})
    atomic_json(out/('completed.json' if step==200 else 'paused.json'),{'step':step,'seconds':time.monotonic()-start,'initial_miou':initial['summary']['all']['miou'],'final_miou':final['summary']['all']['miou'],'delta':final['summary']['all']['miou']-initial['summary']['all']['miou']})


if __name__=='__main__':main()
