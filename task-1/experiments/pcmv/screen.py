"""One PixelSeal fit/evaluation; two GPU feature workers on one shared node."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments.native_localizer.common import atomic_json


def plan(root,out):
    from experiments.triforensic_wm.data import Pool
    from experiments.native_localizer.generator import EncodedDataset
    from wmloc.data import LocalisationDataset
    split=json.loads((root/'triforensic-runs/pixelseal/unified-v1/split.json').read_text())
    rng=np.random.default_rng(20261122);train=sorted(split['train']);rng.shuffle(train);cal=set(train[:23]);valid=set(split['validation'])
    ds=LocalisationDataset('data','train',methods=['pixelseal']);tasks=[]
    for i in range(len(ds)):
        n=ds[i]['image_name'];role='validation' if n in valid else 'calibration' if n in cal else 'train'
        tasks.append({'kind':'real','index':i,'id':n,'role':role})
    enc=EncodedDataset(root/'encoded/pixelseal/pilot-train/encoded.jsonl','pixelseal');unique={r['source_id']:i for i,r in enumerate(enc.rows)};chosen=list(unique.values())[:64];reserved={enc.rows[i]['source_id'] for i in chosen}
    pool=Pool(root/'triforensic-data/pixelseal/train');eligible=[i for i,r in enumerate(pool.rows) if not {r['source_id'],r['background_source_id']}&reserved];rng.shuffle(eligible);counts=[0]*5
    for i in eligible:
        bucket=pool.rows[i]['bucket'];target=pool.get(i,'target')
        if counts[bucket]>=52 or not 0<float(target.mean())<1:continue
        counts[bucket]+=1;tasks.append({'kind':'pool','index':i,'id':f'synthetic-{i}','role':'train'})
        if sum(counts)==260:break
    for j,i in enumerate(chosen):
        for wm in (0,1):tasks.append({'kind':'control','partition':'train','index':i,'variant':j,'marked':wm,'id':f'train-control-{j}-{wm}','role':'train' if j<32 else 'calibration_control'})
    dev=EncodedDataset(root/'encoded/pixelseal/pilot-dev/encoded.jsonl','pixelseal');unique_dev=list({r['source_id']:i for i,r in enumerate(dev.rows)}.values())
    assert not set(unique)&{r['source_id'] for r in dev.rows}
    for j in range(200):
        for wm in (0,1):tasks.append({'kind':'control','partition':'dev','index':unique_dev[j%len(unique_dev)],'variant':j,'marked':wm,'id':f'dev-control-{j}-{wm}','role':'audit_control'})
    atomic_json(out/'plan.json',{'tasks':tasks,'seed':20261122,'real_train':90,'real_calibration':23,'real_validation':57,'synthetic_partial':sum(counts),'synthetic_buckets':counts,'anti_shortcut_views':400,'anti_shortcut_distinct_sources':len(unique_dev),'note':'200 realization pairs from128 distinct dev sources; additional message variants repeat sources. One reused developmental fold; no validation GT in features/proposals/calibration.'})


def worker(root,out,rank,world):
    import torch
    from experiments.triforensic_wm.model import TriForensicWM
    from experiments.message_localizer.model import MessageLocalizer
    from experiments.triforensic_wm.data import Pool
    from experiments.native_localizer.generator import EncodedDataset
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from .base_candidate import GlobalCarrierBase
    from .proposals import Proposals,action_targets
    from .features import Evidence,tensor
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    candidate=json.loads((root/'pcmv-runs/base-candidate/evaluation.json').read_text());decision=root/'pcmv-runs/base-candidate/decision.json';use_tri=json.loads(decision.read_text())['selected'] if decision.exists() else candidate['selected_for_test']
    tri=TriForensicWM('pixelseal',checkpointing=False).cuda();s=torch.load(root/'triforensic-runs/pixelseal/unified-v1/resume.pt',map_location='cpu',weights_only=False);tri.load_state_dict(s['ema'],strict=True);del s;tri.eval().requires_grad_(False);global_model=GlobalCarrierBase(tri)
    base=None
    if not use_tri:
        base=MessageLocalizer('pixelseal').cuda();s=torch.load(root/'local-presence-runs/ps-data-small/last.pt',map_location='cpu',weights_only=False);base.load_state_dict(s['model'],strict=True);del s;base.eval().requires_grad_(False)
    wm=load_watermark('pixelseal',models_dir='checkpoints',device='cuda');wm.model.requires_grad_(False)
    sources={part:EncodedDataset(root/f'encoded/pixelseal/pilot-{part}/encoded.jsonl','pixelseal') for part in ('train','dev')}
    bias=[]
    with torch.inference_mode():
        for i in range(32):bias.append(wm.logits(tensor(sources['train'].load(i)[0])[None])[0].cpu().numpy())
    evidence=Evidence(wm,np.mean(bias,axis=0));proposal=Proposals(root/'research-weights/sam2/sam2.1_hiera_small.pt')
    pool=Pool(root/'triforensic-data/pixelseal/train');ds=LocalisationDataset('data','train',methods=['pixelseal']);tasks=json.loads((out/'plan.json').read_text())['tasks'];dest=out/'features';dest.mkdir(exist_ok=True)
    with np.load(root/'pcmv-runs/base-candidate/all-real-maps.npz') as archive:real={k:archive[k] for k in ('names','probabilities','local_logits','observability')}
    lookup={str(n):i for i,n in enumerate(real['names'])};start=time.monotonic();done=0
    for task in tasks[rank::world]:
        path=dest/(task['id']+'.npz')
        if path.exists():continue
        if task['kind']=='real':
            it=ds[task['index']];rgb=it['image'];gt=it['mask128'];bits=it['message'];source=it['image_name']
        elif task['kind']=='pool':
            i=task['index'];r=pool.rows[i];rgb=np.array(pool.get(i,'rgb'));gt=np.array(pool.get(i,'target'));bits=np.array([int(c) for c in r['message']],np.float32);source=r['source_id']
        else:
            clean,_,r=sources[task['partition']].load(task['index']);rgb=clean.copy();source=r['source_id'];rng=np.random.default_rng(731+task['variant']);bits=rng.integers(2,size=256).astype(np.float32)
            # Message variation occurs before exact unchanged embedding.
            if task['marked']:rgb=wm.embed(rgb,bits)
            gt=np.full((128,128),task['marked'],bool)
        with torch.inference_mode():
            x=tensor(rgb)[None];b=torch.tensor(bits,device='cuda')[None].float()
            if task['kind']=='real':
                i=lookup[task['id']];p=real['probabilities'][i];local=real['local_logits'][i];obs=real['observability'][i]
            else:
                d=global_model(x,b);p=d['p'][0].cpu().numpy();local=d['local'][0].cpu().numpy();obs=d['obs'][0].cpu().numpy()
            if base is not None:p=base(x,b)['logits'][0].float().sigmoid().cpu().numpy()
            proposals=proposal(rgb,p,local,obs,limit=48)
            features,wrong,keys,basic=evidence(rgb,bits,proposals,p,local,obs)
        masks=np.stack([r[1] for r in proposals]);target=action_targets(p>=.5,masks,gt.astype(bool))
        np.savez_compressed(path,features=features,wrong=wrong,masks=np.packbits(masks,axis=-1),gains=target,base=p.astype(np.float32),gt=gt.astype(bool),local=local.astype(np.float32),obs=obs.astype(np.float32))
        atomic_json(path.with_suffix('.json'),dict(task,source_id=str(source),proposal_sources=[r[2] for r in proposals],feature_keys=keys,basic_indices=basic,base='TriForensic global1561' if use_tri else 'PS data-small150'))
        done+=1
        if done%5==0:print('PCMV_FEATURES',rank,done,'seconds',round(time.monotonic()-start,1),'last',task['id'],flush=True)
    atomic_json(out/f'worker-{rank}.json',{'seconds':time.monotonic()-start,'completed':done,'rank':rank})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int);p.add_argument('--world',type=int,default=2,choices=[1,2]);a=p.parse_args();out=a.root/'pcmv-runs/screen-v1';out.mkdir(parents=True,exist_ok=True)
    if a.worker is not None:worker(a.root,out,a.worker,a.world);return
    if not (out/'plan.json').exists():plan(a.root,out)
    assert not (out/'evaluation.json').exists(),'Finished experiment; do not silently refit'
    processes=[];handles=[]
    for rank in range(a.world):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1');handle=(out/f'worker-{rank}.log').open('a');handles.append(handle)
        processes.append(subprocess.Popen([sys.executable,'-u','-m','experiments.pcmv.screen','--root',str(a.root),'--worker',str(rank),'--world',str(a.world)],env=env,stdout=handle,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for h in handles:h.close()
    assert codes==[0]*a.world,codes
    from .fit import fit
    fit(out)


if __name__=='__main__':main()
