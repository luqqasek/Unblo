"""Restore150 optimizer/RNG and continue300 updates; preserve historical maps for matched audit."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from experiments.local_presence.data import full_examples,down
from experiments.triforensic_wm.evaluate import summarize as summary
from experiments.local_presence.verification import verified
from experiments.native_localizer.common import seed_all,atomic_json,append_jsonl,rgb_tensor,rng_state,restore_rng,sha256
from experiments.native_localizer.generator import rect_mask,rectangle
from experiments.native_localizer.losses import segmentation
from experiments.native_localizer.evaluate import image_metrics
from experiments.message_localizer.model import MessageLocalizer
from experiments.region_data.data import RegionData
from experiments.small_zoom.proposals import boxmask
from wmloc.data import LocalisationDataset

class Replay:
    def __init__(self,root,family,online):
        self.online=online;self.root=root;self.family=family;self.used=[]
        if online:
            self.data=RegionData(root,family,'train');self.folder=root/f'local-presence-data/{family}/train';self.tiny=json.loads((self.folder/'manifest.json').read_text())
        else:self.fixed=list(full_examples(root,family,False,128))
    def sample(self,rng):
        if not self.online:
            i=int(rng.integers(len(self.fixed)));self.used.append({'fixed_index':i});return self.fixed[i]
        if rng.random()<.5:
            xs,ys,meta=self.data.sample(rng,novel=True,paired=False);m=meta[0];self.used.append(m);return xs[0],ys[0],self.data.messages[m['realization_id']]
        if rng.random()<.5:
            row=self.tiny[int(rng.integers(len(self.tiny)))];clean,_,old=self.data.old.load(row['index']);marked=clean.copy();x0,y0,x1,y1=row['rectangle'];marked[y0:y1,x0:x1]=np.asarray(Image.open(self.folder/row['file']));support=rect_mask(row['rectangle']);kind='tiny_crop'
        else:
            clean,marked,old=self.data.old.load(int(rng.choice(self.data.full_indices)));support=rect_mask(rectangle(rng,float(np.exp(rng.uniform(np.log(.0025),np.log(.05)))),16));kind='large_source_remnant'
            if rng.random()<.5:support=~support;kind='same_content_clean_hole'
        x=np.where(support[...,None],marked,clean).astype(np.uint8);y=down(support)
        self.used.append({'source_id':old['source_id'],'realization_id':old['realization_id'],'operation':kind,'fg_fraction':float(y.mean()),'initial_rectangle':row['rectangle'] if kind=='tiny_crop' else old['rectangle']})
        return x,y,old['message']

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',choices=['pixelseal','trustmark'],required=True);p.add_argument('--online',action='store_true');a=p.parse_args()
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;seed_all(4242)
    prefix='ps' if a.family=='pixelseal' else 'tm';name=prefix+'-data-'+('online' if a.online else 'small');parent=a.root/'local-presence-runs'/name;out=a.root/'message-extended-runs'/name;out.mkdir(parents=True,exist_ok=True);assert not (out/'completed.json').exists();started=time.monotonic()
    print('DATA_CONTINUATION_START',name,flush=True);replay=Replay(a.root,a.family,a.online)
    ds=LocalisationDataset('data','train',methods=[a.family]);split=json.loads(Path('experiments/native_localizer/reconstructed_splits.json').read_text())['families'][a.family][0];valid=[i for i,r in enumerate(ds.meta) if r['image_name'] in split['validation']];train=[i for i in range(len(ds)) if i not in valid];real={i:ds[i] for i in train};rng=np.random.default_rng(4242);realrng=np.random.default_rng(12342)
    recipe='control-adapt' if a.family=='pixelseal' else 'objects-adapt';path=a.root/f'region-runs/{a.family}/{recipe}/fold1/model.pt';model=MessageLocalizer(a.family).cuda();state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);del state;model.freeze_encoder(False);model.train()
    enc=list(model.encoder.net.parameters());ids={id(p) for p in enc};new=[p for p in model.parameters() if id(p) not in ids and p.requires_grad];opt=torch.optim.AdamW([{'params':enc,'lr':2e-6},{'params':new,'lr':2e-5}],weight_decay=.01)
    rects={r['image_name']:r for r in json.loads((a.root/'small-zoom-finalscale/trustmark/proposals.json').read_text())} if a.family=='trustmark' else {}
    config={'family':a.family,'online':a.online,'steps':450,'effective_batch':16,'synthetic_views':12,'real_views':4,'seed':4242,'fold':1,'threshold':.5,'encoder_lr':2e-6,'new_lr':2e-5,'all_localizer_trainable':True,'BN_running_stats':'eval','base':str(path),'precision':'FP32, TF32off','new_embedding_calls':0,'policy':'50% RegionData online objects/legacy;25% exact tiny crops;25% full-source remnant/hole' if a.online else '128 fixed full_examples composites reused','endpoint_fixed':450,'continued_from':str(parent/'last.pt'),'parent_sha256':sha256(parent/'last.pt'),'incremental_updates':300,'selection':'fixed450 vs150 and original0; >=.005 mIoU gain, no>.01 tiny-FG or hole regression; exploratory fold1'}
    atomic_json(out/'split.json',{'train':[ds.meta[i]['image_name'] for i in train],'validation':split['validation']});atomic_json(out/'config.json',config)
    def evaluate(step):
        model.eval();rows=[];probs=[];ys=[];names=[];saferows=[];safeprobs=[];rawprobs=[]
        with torch.no_grad():
            for i in valid:
                it=ds[i];x=rgb_tensor(it['image'])[None].cuda();b=torch.tensor(it['message'])[None].cuda().float();prob=model(x,b)['logits'][0].sigmoid().cpu().numpy();n=it['image_name'];cand=rects.get(n,{}).get('correct_top',[])
                rawprobs.append(prob.copy());safe=prob.copy()
                if cand and verified(cand[0]):
                    box=cand[0]['box'];area=(box[2]-box[0])*(box[3]-box[1])/512**2
                    if area<=.15:safe[boxmask(box)]=1
                if cand and cand[0]['bit_accuracy']>=.9:prob[boxmask(cand[0]['box'])]=1
                sr=image_metrics(safe,it['mask128']);sr['image_name']=n;saferows.append(sr);safeprobs.append(safe)
                r=image_metrics(prob,it['mask128']);r['image_name']=n;rows.append(r);probs.append(prob);ys.append(it['mask128']);names.append(n)
        atomic_json(out/f'correction-{step}.json',{'summary':summary(rows),'per_image':rows,'safe_summary':summary(saferows),'safe_per_image':saferows,'safe_policy':'integer90/100 gate; accepted rectangles <=15%; no new decoder calls'});np.savez_compressed(out/f'maps-{step}.npz',names=names,probabilities=probs,targets=ys,safe_probabilities=safeprobs,raw_probabilities=rawprobs);model.train();print('DATA_VALIDATION',name,step,json.dumps(summary(rows)),flush=True);return rows
    state=torch.load(parent/'last.pt',map_location='cpu',weights_only=False)
    assert state['step']==150 and state['config']['family']==a.family and state['config']['online']==a.online
    assert json.loads((parent/'split.json').read_text())==json.loads((out/'split.json').read_text())
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    rng.bit_generator.state=state['data_rng'];realrng.bit_generator.state=state['real_rng'];restore_rng(state['rng']);del state
    baseline=evaluate(150)
    old=np.load(parent/'maps-150.npz');now=np.load(out/'maps-150.npz')
    assert np.array_equal(old['names'],now['names']) and np.array_equal(old['targets'],now['targets'])
    error=float(np.max(np.abs(old['probabilities']-now['probabilities'])))
    atomic_json(out/'reproduction.json',{'max_abs_probability_difference':error,'tolerance':1e-4})
    assert error<1e-4,('Parent reproduction mismatch',error)
    micro=16;training_start=time.monotonic()
    for step in range(150,450):
        examples=[replay.sample(rng) for _ in range(12)];items=[real[int(i)] for i in realrng.choice(train,4,replace=True)]
        x=torch.stack([rgb_tensor(v[0]) for v in examples]+[rgb_tensor(it['image']) for it in items]).cuda();y=torch.stack([torch.tensor(v[1]).float() for v in examples]+[torch.tensor(it['mask128']).float() for it in items]).cuda();bits=torch.tensor([list(v[2]) for v in examples]+[it['message'].tolist() for it in items],device='cuda').float()
        def backward(microbatch):
            value=0.;opt.zero_grad(set_to_none=True)
            for j in range(0,len(x),microbatch):
                loss=segmentation(model(x[j:j+microbatch],bits[j:j+microbatch])['logits'],y[j:j+microbatch])*(min(microbatch,len(x)-j)/len(x));loss.backward();value+=float(loss.detach())
            return value
        oom=False
        try:value=backward(micro)
        except torch.cuda.OutOfMemoryError:
            if step!=150 or micro!=16:raise
            oom=True
        if oom:
            opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();micro=8;value=backward(micro);print('MICROBATCH_FALLBACK',name,8,flush=True)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
        if step==150 or (step+1)%10==0:
            seconds=time.monotonic()-training_start;row={'step':step+1,'loss':value,'gradient_norm':float(norm),'microbatch':micro,'seconds':seconds,'seconds_per_step':seconds/(step+1-150),'peak_bytes':torch.cuda.max_memory_allocated(),'estimated_remaining_seconds':seconds/(step+1-150)*(450-step-1)};append_jsonl(out/'train.jsonl',row);print('DATA_PROGRESS',name,json.dumps(row),flush=True)
        if step+1 in [250,450]:
            final=evaluate(step+1);torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step+1,'config':config,'rng':rng_state(),'data_rng':rng.bit_generator.state,'real_rng':realrng.bit_generator.state},out/'last.pt')
    atomic_json(out/'replay_exposure.json',{'sampled_views':len(replay.used),'unique_fixed_indices':len({r['fixed_index'] for r in replay.used if 'fixed_index' in r}),'unique_sources':len({r['source_id'] for r in replay.used if 'source_id' in r}),'operations':{op:sum(r.get('operation')==op for r in replay.used) for op in sorted({r.get('operation','fixed') for r in replay.used})},'recipes':replay.used})
    atomic_json(out/'completed.json',{'baseline':summary(baseline),'final':summary(final),'delta_miou':float(np.mean([r['miou'] for r in final])-np.mean([r['miou'] for r in baseline])),'seconds':time.monotonic()-started,'peak_bytes':torch.cuda.max_memory_allocated(),'microbatch':micro,'optimizer_updates':300,'views':4800})
    print('DATA_DONE',name,(out/'completed.json').read_text(),flush=True)
if __name__=='__main__':main()
