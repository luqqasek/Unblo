"""Matched continuation: same model/init/budget, old vs object/remnant data."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.ndimage import label,distance_transform_edt,binary_erosion
import torch
from experiments.native_localizer.common import atomic_json,append_jsonl,seed_all,rgb_tensor,rng_state
from experiments.native_localizer.evaluate import image_metrics,summarize,gallery
from experiments.native_localizer.losses import segmentation,pair_loss
from experiments.message_localizer.model import MessageLocalizer
from experiments.message_localizer.extractor import Extractor
from experiments.message_localizer.train import evaluate
from wmloc.data import LocalisationDataset
from .prepare import prepare
from .data import RegionData

def geometry(mask,image):
    m=mask.astype(bool);minor=m if m.mean()<=.5 else ~m
    yy,xx=np.nonzero(minor);dist=distance_transform_edt(minor)
    bbox=(xx.max()-xx.min()+1)*(yy.max()-yy.min()+1) if len(xx) else 0
    gray=image.astype(np.float32).mean(2);edge=(abs(np.diff(gray,axis=0)).mean()+abs(np.diff(gray,axis=1)).mean())/2
    boundary=minor & ~binary_erosion(minor)
    return {'fg_fraction':float(m.mean()),'minority_fraction':float(minor.mean()),'bbox_fill':float(minor.sum()/bbox) if bbox else None,'components':int(label(minor)[1]),'width_p50_grid':float(np.median(2*dist[minor])) if minor.any() else None,'boundary_to_area':float(boundary.sum()/max(1,minor.sum())),'texture_gradient_rgb_units':float(edge)}

def diagnose(data,family,root,out):
    rows=[];extractor=Extractor(family).cuda().eval();extractor.requires_grad_(False)
    for variant in ['control','objects']:
        rng=np.random.default_rng(991)
        for i in range(128):
            xs,ys,meta=data.sample(rng,variant=='objects',paired=True)
            row=dict(meta[0],variant=variant,**geometry(ys[0],xs[0]));row['paired_geometry_same']=True
            if i<64:
                with torch.inference_mode():
                    _,bits=extractor(rgb_tensor(xs[0])[None].cuda(),extractor.native)
                    msg=np.asarray(data.messages[meta[0]['realization_id']]);z=bits.mean((-2,-1))[0].cpu().numpy()
                    row['global_bit_accuracy']=float(((z>0)==msg).mean())
                # For novel clean twins this is the exact same-content residual, not inferred support.
                if meta[0]['pair_kind']=='clean_twin':
                    residual=xs[0].astype(np.float32)-xs[1].astype(np.float32)
                    region=np.repeat(np.repeat(ys[0].astype(bool),4,0),4,1)
                    if region.any():
                        row['residual_rms_fg_cells']=float(np.sqrt((residual[region]**2).mean()))
                        row['zero_pixel_fraction_fg_cells']=float(np.all(residual[region]==0,axis=1).mean())
            rows.append(row)
    del extractor;torch.cuda.empty_cache()
    atomic_json(out/'synthetic_geometry_signal.json',rows)
    real=LocalisationDataset('data','train',methods=[family]);atomic_json(out/'real_geometry.json',[dict(image_name=x['image_name'],**geometry(x['mask128'],x['image'])) for x in real])
    manifests=[dict(realization_id=r['realization_id'],source_id=r['source_id'],rectangle=r['rectangle'],regime=r['regime'],residual_uint8=r['residual_uint8']) for r in data.rows]
    atomic_json(out/'encoding_signal.json',manifests)
    print('DIAGNOSTICS_COMPLETE',family,flush=True)

@torch.inference_mode()
def eval_objects(model,data,out):
    out.mkdir(parents=True,exist_ok=True);model.eval();rng=np.random.default_rng(2099);rows=[]
    for i in range(64):
        xs,ys,meta=data.sample(rng,True,False);m=meta[0]
        z=model(rgb_tensor(xs[0])[None].cuda(),torch.tensor(data.messages[m['realization_id']],device='cuda')[None].float())['logits']
        prob=z.sigmoid()[0].cpu().numpy();r=image_metrics(prob,ys[0],.5);r.update(image_name=f'object_dev_{i}',regime=m['regime']);rows.append(r)
        if i<4:gallery(out/f'example_{i}.png',xs[0],ys[0],prob)
    atomic_json(out/'per_image.json',rows);atomic_json(out/'metrics.json',summarize(rows));model.train()
    return summarize(rows)['groups']['all']['miou']['mean']

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--steps',type=int,default=200);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'region-runs'/a.family;out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'protocol.json',{'steps':a.steps,'batch':32,'seed':42,'encoder_lr':5e-6,'new_lr':5e-5,'init':'same400-step synthetic checkpoint','loss':'unchanged per-image segmentation +0.1 pair ranking','threshold':.5,'real_training':False,'change':'generator only; 50% legacy/25% same-content COCO/25% cross-image COCO; new-branch minority bins35/35/20/10, equal polarity; 25% thin remnants','no_post_embedding_resampling':True})
    prepare(a.root,a.family)
    data=RegionData(a.root,a.family,'train');dev=RegionData(a.root,a.family,'dev',preload=False)
    assert not {r['source_id'] for r in data.rows}&{r['source_id'] for r in dev.rows}
    atomic_json(out/'data_validation.json',{'train_source_count':len({r['source_id'] for r in data.rows}),'dev_source_count':len({r['source_id'] for r in dev.rows}),'train_object_masks':len(data.pool),'dev_object_masks':len(dev.pool),'source_disjoint':True,'train_new_embedding_calls':128,'dev_new_embedding_calls':32,'pool_bin_counts':{str(k):len(v) for k,v in data.bybin.items()}})
    diagnose(dev,a.family,a.root,out)
    state=torch.load(a.root/f'message-runs/{a.family}/matched-v1/last.pt',map_location='cpu',weights_only=False)
    for variant in ['control','objects']:
        dest=out/variant;dest.mkdir(exist_ok=True)
        if (dest/'completed.json').exists():continue
        seed_all(42);rng=np.random.default_rng(42);model=MessageLocalizer(a.family).cuda();model.load_state_dict(state['model']);model.freeze_encoder(False);model.train();forward=torch.nn.DataParallel(model,device_ids=[0,1])
        enc=list(model.encoder.net.parameters());ids={id(p) for p in enc};new=[p for p in model.parameters() if id(p) not in ids and p.requires_grad]
        opt=torch.optim.AdamW([{'params':enc,'lr':5e-6},{'params':new,'lr':5e-5}],weight_decay=.01);started=time.monotonic();counts=Counter()
        start_step=0
        if (dest/'last.pt').exists():
            resumed=torch.load(dest/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(resumed['model']);start_step=resumed['step']
            assert start_step==a.steps,'Only completed training checkpoints are resumed in this pilot'
            if 'effective_views' in resumed:counts.update(resumed['effective_views'])
            else:
                # Recreate exact sampling counts for the completed pre-fix control, without retraining.
                for _ in range(start_step):
                    for group in range(24):
                        _,_,ms=data.sample(rng,variant=='objects',group<8)
                        for m in ms:
                            f=m['fg_fraction'];counts['views']+=1;counts[m['operation']]+=1;counts['fg_le1']+=int(0<f<=.01);counts['fg_1to5']+=int(.01<f<=.05);counts['clean_le1']+=int(.99<=f<1);counts['clean_1to5']+=int(.95<=f<.99);counts['empty']+=int(f==0);counts['full']+=int(f==1)
            del resumed
            print('RESUME_EVALUATION_ONLY',a.family,variant,flush=True)
        for step in range(start_step,a.steps):
            x,y,pairs,meta=data.batch(rng,32,variant=='objects');b=torch.tensor([data.messages[m['realization_id']] for m in meta],device='cuda').float()
            opt.zero_grad(set_to_none=True);z=forward(x.cuda(),b)['logits'];loss=segmentation(z,y.cuda())+.1*pair_loss(z,y.cuda(),pairs);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            opt.step()
            for m in meta:
                f=m['fg_fraction'];counts['views']+=1;counts[m['operation']]+=1;counts['fg_le1']+=int(0<f<=.01);counts['fg_1to5']+=int(.01<f<=.05);counts['clean_le1']+=int(.99<=f<1);counts['clean_1to5']+=int(.95<=f<.99);counts['empty']+=int(f==0);counts['full']+=int(f==1)
            if step%25==0:
                row={'variant':variant,'family':a.family,'step':step+1,'loss':float(loss.detach()),'seconds':time.monotonic()-started};print(json.dumps(row),flush=True);append_jsonl(dest/'train.jsonl',row)
            if step<4 or (step+1)%25==0:append_jsonl(dest/'sampled_provenance.jsonl',{'step':step+1,'views':meta})
        if start_step==0:torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':a.steps,'config':{'family':a.family,'variant':variant},'rng':rng_state(),'data_rng':rng.bit_generator.state,'effective_views':dict(counts)},dest/'last.pt')
        olddev=evaluate(model,a.family,a.root,dest/'legacy_dev',dev=True,limit=64);objdev=eval_objects(model,dev,dest/'object_dev');real=evaluate(model,a.family,a.root,dest/'real',dev=False)
        atomic_json(dest/'completed.json',{'seconds_this_invocation':time.monotonic()-started,'resumed_evaluation':bool(start_step),'steps':a.steps,'effective_views':dict(counts),'old_dev_miou':olddev,'object_dev_miou':objdev,'real_miou':real})
        print(json.dumps({'COMPLETE':True,'family':a.family,'variant':variant,'old_dev':olddev,'object_dev':objdev,'real':real}),flush=True)
        del model,forward,opt;torch.cuda.empty_cache()
if __name__=='__main__':main()
