"""One-fold local branch screening; no validation-driven threshold/checkpoint selection."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import Presence,Correction,balanced,soft_iou
from .data import PatchBank,prepare,full_examples
from experiments.native_localizer.common import atomic_json,append_jsonl,seed_all,rgb_tensor
from experiments.native_localizer.evaluate import image_metrics,gallery
from experiments.message_localizer.model import MessageLocalizer
from wmloc.data import LocalisationDataset

def summary(rows):
    groups={'all':rows,'fg_le1':[r for r in rows if 0<r['true_area']<=.01],
      'fg_1to5':[r for r in rows if .01<r['true_area']<=.05], 'fg_5to20':[r for r in rows if .05<r['true_area']<=.2], 'fg_gt20':[r for r in rows if r['true_area']>.2],
      'clean_le1':[r for r in rows if 0<1-r['true_area']<=.01], 'clean_1to5':[r for r in rows if .01<1-r['true_area']<=.05],
      'wm_over50':[r for r in rows if .5<r['true_area']<1]}
    return {k:{'n':len(rr),**{key:float(np.mean([r[key] for r in rr if r.get(key) is not None])) if any(r.get(key) is not None for r in rr) else None for key in ['miou','fg_iou','clean_iou','fg_ap','clean_ap','roc_auc']},'fg_miss':sum(r['fg_iou']==0 for r in rr),'clean_miss':sum(r['clean_iou']==0 for r in rr)} for k,rr in groups.items()}

@torch.no_grad()
def local_eval(model,ds,valid,out):
    model.eval();rows=[]
    for i in valid:
        item=ds[i];z,_=model(rgb_tensor(item['image'])[None].cuda());r=image_metrics(z[0].sigmoid().cpu().numpy(),item['mask128']);r['image_name']=item['image_name'];rows.append(r)
    s=summary(rows);atomic_json(out,{'summary':s,'per_image':rows});model.train();return s

@torch.no_grad()
def synthetic_eval(model,bank,out):
    model.eval();rows=[];from sklearn.metrics import roc_auc_score
    for start in range(0,min(256,len(bank.images)),32):
        x=bank.images[start:start+32].flatten(0,1).cuda().float()/255;y=bank.targets[start:start+32].flatten(0,1).numpy();z,_=model(x);prob=z.sigmoid().cpu().numpy()
        for j,(p,t) in enumerate(zip(prob,y)):
            r=image_metrics(p,t);r['regime']=bank.meta[start+j//2]['regime'];rows.append(r)
    s={g:summary([r for r in rows if g=='all' or r['regime']==g])['all'] for g in ['all',*sorted({r['regime'] for r in rows})]}
    atomic_json(out,s);model.train();return s

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--holes',action='store_true');p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    torch.set_num_threads(6);seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    name=a.family+('-holes' if a.holes else '-presence');out=a.root/'local-presence-runs'/name;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'completed.json').exists(),'Preserve completed runs'
    start=time.monotonic()
    if not a.holes:prepare(a.root,a.family)
    train=PatchBank(a.root,a.family,holes=a.holes,count=2048);dev=PatchBank(a.root,a.family,'dev',holes=a.holes,count=256)
    atomic_json(out/'patch_provenance.json',{'train':train.meta,'dev':dev.meta})
    assert not {r['source_id'] for r in train.meta}&{r['source_id'] for r in dev.meta}
    model=Presence().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);rng=np.random.default_rng(42)
    ds=LocalisationDataset('data','train',methods=[a.family]);lookup={r['image_name']:i for i,r in enumerate(ds.meta)}
    split=json.loads(Path('experiments/native_localizer/reconstructed_splits.json').read_text())['families'][a.family][0]
    valid=[lookup[n] for n in split['validation']];training=[i for i in range(len(ds)) if i not in valid]
    atomic_json(out/'config.json',{'family':a.family,'holes':a.holes,'patch_steps':a.steps,'patch_batch':128,'seed':42,'fold':1,'correction_steps':300,'threshold':.5,'rf':27,'params':sum(p.numel() for p in model.parameters()),'precision':'FP32 raw, stem, loss; TF32 off','norm':'none','data':'exact local crop, full-source remnant, clean restoration; no post-embed resampling','observability':'local high-pass magnitude proxy, not original JND in this auxiliary branch','selection':'fixed1000+300 endpoint, no validation threshold tuning'})
    atomic_json(out/'split.json',{'train':[ds.meta[i]['image_name'] for i in training],'validation':split['validation']})
    trainstart=time.monotonic()
    for step in range(a.steps):
        x,y=train.batch(rng);z,_=model(x)
        changed=y[0::2]!=y[1::2];direction=torch.sign(y[1::2]-y[0::2]);rank=F.softplus(1-direction*(z[1::2]-z[0::2]))
        pair=rank[changed].mean() if changed.any() else z.sum()*0
        patch=F.binary_cross_entropy_with_logits(torch.logsumexp(z.flatten(1),1)-np.log(z.shape[-1]*z.shape[-2]),y.flatten(1).amax(1))
        loss=.5*balanced(z,y)+.5*F.binary_cross_entropy_with_logits(z,y)+.1*pair+.05*patch
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
        if (step+1)%100==0:
            row={'step':step+1,'loss':float(loss),'pair':float(pair),'seconds':time.monotonic()-start,'train_seconds':time.monotonic()-trainstart,'peak_bytes':torch.cuda.max_memory_allocated()};append_jsonl(out/'train.jsonl',row);print(name,json.dumps(row),flush=True)
        if step+1 in [250,a.steps]:
            syn=synthetic_eval(model,dev,out/f'synthetic-{step+1}.json');real=local_eval(model,ds,valid,out/f'local-real-{step+1}.json')
            print('LOCAL_VALIDATION',name,step+1,json.dumps(real),flush=True)
            torch.save({'model':model.state_dict(),'step':step+1,'optimizer':opt.state_dict()},out/'patch-resume.pt')
    model.eval().requires_grad_(False);del opt;torch.cuda.empty_cache()
    variant='control' if a.family=='pixelseal' else 'objects'
    checkpoint=a.root/f'region-runs/{a.family}/{variant}-adapt/fold1/model.pt'
    base=MessageLocalizer(a.family).cuda();state=torch.load(checkpoint,map_location='cpu',weights_only=False);base.load_state_dict(state['model']);del state;base.eval().requires_grad_(False)
    feats=[];logits=[];targets=[]
    with torch.no_grad():
        for i in range(len(ds)):
            item=ds[i];x=rgb_tensor(item['image'])[None].cuda();b=torch.tensor(item['message'])[None].cuda().float()
            _,f=model(x);z=base(x,b)['logits'];feats.append(f[0].half().cpu());logits.append(z[0].cpu());targets.append(torch.tensor(item['mask128']).float())
    features=torch.stack(feats).cuda();zs=torch.stack(logits).cuda();ys=torch.stack(targets).cuda();del feats,logits,targets
    sf=[];sz=[];sy=[]
    with torch.no_grad():
        for image,y,bits in full_examples(a.root,a.family,a.holes):
            x=rgb_tensor(image)[None].cuda();b=torch.tensor(bits)[None].cuda().float();_,f=model(x);z=base(x,b)['logits']
            sf.append(f[0].half().cpu());sz.append(z[0].cpu());sy.append(torch.tensor(y))
    synf=torch.stack(sf).cuda();synz=torch.stack(sz).cuda();syny=torch.stack(sy).cuda();del base,sf,sz,sy;torch.cuda.empty_cache()
    correction=Correction(a.holes).cuda();opt=torch.optim.AdamW(correction.parameters(),lr=1e-3,weight_decay=.01)
    corrections={}
    if a.family=='trustmark':
        from experiments.small_zoom.proposals import boxmask
        for row in json.loads((a.root/'small-zoom-finalscale/trustmark/proposals.json').read_text()):
            if row['correct_top'] and row['correct_top'][0]['bit_accuracy']>=.9:corrections[row['image_name']]=boxmask(row['correct_top'][0]['box'])
    def evaluate(step):
        rows=[];probs=[];ysout=[];names=[]
        with torch.no_grad():
            for i in valid:
                prob=correction(zs[i:i+1],features[i:i+1].float())[0].sigmoid().cpu().numpy();n=ds.meta[i]['image_name']
                if n in corrections:prob[corrections[n]]=1
                y=ys[i].cpu().numpy();r=image_metrics(prob,y);r['image_name']=n;rows.append(r);probs.append(prob);ysout.append(y);names.append(n)
        atomic_json(out/f'correction-{step}.json',{'summary':summary(rows),'per_image':rows})
        if step in [0,300]:np.savez_compressed(out/f'maps-{step}.npz',names=names,probabilities=np.array(probs),targets=np.array(ysout))
        print('CORRECTION_VALIDATION',name,step,json.dumps(summary(rows)),flush=True)
        return rows
    baseline=evaluate(0);realrng=np.random.default_rng(12342)
    for step in range(300):
        idx=realrng.choice(training,8,replace=True);zr=correction(zs[idx],features[idx].float());loss=.5*balanced(zr,ys[idx])+.5*soft_iou(zr,ys[idx])
        # 75% synthetic replay: actual base predictions and local features of full512 images.
        si=rng.integers(len(synz),size=24);corrected=correction(synz[si],synf[si].float());y=syny[si]
        synthetic=.5*balanced(corrected,y)+.5*soft_iou(corrected,y)
        loss=.25*loss+.75*synthetic+.001*(zr-zs[idx]).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(correction.parameters(),1);opt.step()
        if step+1 in [100,300]:final=evaluate(step+1)
    torch.save({'local':model.state_dict(),'correction':correction.state_dict(),'family':a.family,'holes':a.holes,'base_checkpoint':str(checkpoint)},out/'last.pt')
    atomic_json(out/'completed.json',{'seconds':time.monotonic()-start,'baseline':summary(baseline),'final':summary(final),'delta_miou':float(np.mean([r['miou'] for r in final])-np.mean([r['miou'] for r in baseline])),'peak_bytes':torch.cuda.max_memory_allocated()})
    print('DONE',name,(out/'completed.json').read_text(),flush=True)
if __name__=='__main__':main()
