"""Repair uint8 sign arithmetic before any scorer fitting; cached baseline unchanged."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np


def worker(root,out,rank):
    import torch
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from experiments.native_localizer.generator import EncodedDataset
    from experiments.native_localizer.common import atomic_json
    from .features import Evidence,tensor
    from .proposals import Proposals,action_targets
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    wm=load_watermark('pixelseal',models_dir='checkpoints',device='cuda');wm.model.requires_grad_(False);source=EncodedDataset(root/'encoded/pixelseal/pilot-train/encoded.jsonl','pixelseal');ds=LocalisationDataset('data','train',methods=['pixelseal'])
    with torch.inference_mode():bias=np.mean([wm.logits(tensor(source.load(i)[0])[None])[0].cpu().numpy() for i in range(32)],axis=0)
    evidence=Evidence(wm,bias);proposal=Proposals(root/'research-weights/sam2/sam2.1_hiera_small.pt');tasks=[t for t in json.loads((out/'plan.json').read_text())['tasks'][rank::2] if t['kind']=='real'];start=time.monotonic()
    for j,task in enumerate(tasks):
        path=out/'features'/(task['id']+'.npz');meta=json.loads(path.with_suffix('.json').read_text())
        if meta.get('signed_bits_fixed'):continue
        with np.load(path) as old:data={k:old[k] for k in old.files}
        item=ds[task['index']];bits=item['message'].astype(np.float32);rgb=item['image'];p=data['base'];local=data['local'];obs=data['obs'];proposals=proposal(rgb,p,local,obs,limit=48);masks=np.stack([r[1] for r in proposals]);assert np.array_equal(np.packbits(masks,axis=-1),data['masks']),'Proposal reproduction mismatch'
        f,w,keys,basic=evidence(rgb,bits,proposals,p,local,obs);assert keys==meta['feature_keys'];data['features']=f;data['wrong']=w;temporary=path.with_suffix('.repair.npz');np.savez_compressed(temporary,**data);temporary.replace(path);meta['signed_bits_fixed']=True;atomic_json(path.with_suffix('.json'),meta)
        if (j+1)%10==0:print('PCMV_REPAIR256',rank,j+1,round(time.monotonic()-start,1),flush=True)
    del wm,proposal,evidence;torch.cuda.empty_cache()
    from .native_probe import worker as native_worker
    native_worker(root,out,rank,2,real_only=True)
    atomic_json(out/f'repair-{rank}.json',{'completed':True,'real_images':len(tasks),'seconds':time.monotonic()-start,'no_scorer_trained_before_repair':True,'base_predictions_changed':False});print('PCMV_REPAIR_DONE',rank,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int);a=p.parse_args();out=a.root/'pcmv-runs/screen-v1'
    if a.worker is not None:worker(a.root,out,a.worker);return
    ps=[]
    for rank in range(2):
        h=(out/f'repair-{rank}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1');ps.append(subprocess.Popen([sys.executable,'-u','-m','experiments.pcmv.repair_real','--root',str(a.root),'--worker',str(rank)],env=env,stdout=h,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in ps];assert codes==[0,0],codes
    from .fit import fit
    fit(out,core_only=True)


if __name__=='__main__':main()
