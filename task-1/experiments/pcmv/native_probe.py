"""Native512 ablation supplement justified by archived RTX signal measurements.

This deliberately bypasses the official wrapper resize, unlike crop verification.
It is a separate frozen-decoder diagnostic feature, not an altered embedder.
"""
import argparse,json,os,subprocess,sys,time,fcntl
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

KEYS=['native512_full_z','native512_ablation_z','native512_drop','native512_drop_capacity','native512_masked_pool','native512_jnd_pool','native512_rank_drop']


def worker(root,out,rank,world,real_only=False):
    planned=json.loads((out/'plan.json').read_text())['tasks'][rank::world]
    if not real_only and all((out/'native512'/(t['id']+'.npz')).exists() for t in planned):return
    import torch
    from torch.nn import functional as F
    from wmloc.watermarks import load_watermark
    from wmloc.data import LocalisationDataset
    from experiments.triforensic_wm.data import Pool
    from experiments.native_localizer.generator import EncodedDataset
    from .features import tensor
    from .proposals import ring
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    wm=load_watermark('pixelseal',models_dir='checkpoints',device='cuda');wm.model.requires_grad_(False);source={p:EncodedDataset(root/f'encoded/pixelseal/pilot-{p}/encoded.jsonl','pixelseal') for p in ('train','dev')};pool=Pool(root/'triforensic-data/pixelseal/train');real=LocalisationDataset('data','train',methods=['pixelseal'])
    dest=out/'native512';dest.mkdir(exist_ok=True);tasks=json.loads((out/'plan.json').read_text())['tasks'];start=time.monotonic()
    @torch.inference_mode()
    def decode(images):
        return np.concatenate([wm.det(torch.stack([tensor(im) for im in images[i:i+8]]))[:,1:].float().cpu().numpy() for i in range(0,len(images),8)])
    bias=decode([source['train'].load(i)[0] for i in range(32)]).mean(0)
    done=0
    with torch.inference_mode():
        for task in tasks[rank::world]:
            if real_only and task['kind']!='real':continue
            path=dest/(task['id']+'.npz')
            if path.exists() and not real_only:continue
            # Can stream alongside proposal workers on the SAME two GPUs.
            # JSON is written only after the corresponding NPZ is complete.
            ready=out/'features'/(task['id']+'.json');deadline=time.monotonic()+600
            while not ready.exists():
                assert time.monotonic()<deadline,('Producer stalled',task['id'])
                time.sleep(1)
            if task['kind']=='real':it=real[task['index']];rgb=it['image'];bits=it['message']
            elif task['kind']=='pool':rgb=np.array(pool.get(task['index'],'rgb'));bits=np.array([int(c) for c in pool.rows[task['index']]['message']],np.float32)
            else:
                rgb=source[task['partition']].load(task['index'])[0].copy();bits=np.random.default_rng(731+task['variant']).integers(2,size=256).astype(np.float32)
                if task['marked']:rgb=wm.embed(rgb,bits)
            with np.load(out/'features'/(task['id']+'.npz')) as archive:masks=np.unpackbits(archive['masks'],axis=-1).astype(bool);obs=archive['obs']
            images=[rgb]
            for m in masks:
                r=ring(m);big=np.repeat(np.repeat(m,4,0),4,1);rbig=np.repeat(np.repeat(r,4,0),4,1);fill=np.rint(rgb[rbig].mean(0)).astype(np.uint8);abl=np.where(big[...,None],fill,rgb);assert np.array_equal(abl[~big],rgb[~big]);images.append(abl)
            raw=np.tanh(decode(images)-bias);signs=np.random.default_rng(934).choice([-1.,1.],size=(65,len(bits))).astype(np.float32);signs[0]=np.asarray(bits,dtype=np.float32)*2-1;assert set(np.unique(signs[0])).issubset({-1.,1.});score=raw@signs.T/np.sqrt(np.maximum((raw*raw).sum(1,keepdims=True),1e-12));margin=raw@signs.T/len(bits);cal=(score-score[:,1:].mean(1,keepdims=True))/score[:,1:].std(1,keepdims=True).clip(.05)
            x=wm.det.preprocess(tensor(rgb)[None]);emb=wm.det.pixel_decoder.output_upscaling(wm.det.convnext(x));lin=wm.det.pixel_decoder.linear;dense=F.conv2d(emb,lin.weight[:,:,None,None],lin.bias)[:,1:]
            pooled=[]
            for m in masks:
                w=F.interpolate(torch.tensor(m,device='cuda')[None,None].float(),size=dense.shape[-2:],mode='area');wj=F.interpolate(torch.tensor(m*obs,device='cuda')[None,None].float(),size=dense.shape[-2:],mode='area');pooled.append(torch.cat([(dense*w).sum((2,3))/w.sum().clamp_min(1e-6),(dense*wj).sum((2,3))/wj.sum().clamp_min(1e-6)],0).cpu().numpy())
            v=np.tanh(np.stack(pooled)-bias);ps=np.einsum('kpb,qb->kpq',v,signs)/np.sqrt(np.maximum((v*v).sum(2,keepdims=True),1e-12));capacity=np.array([obs[m].sum()/16384 for m in masks]).clip(1e-5)
            arrays=[]
            for q in (0,1):
                drop=margin[0,q]-margin[1:,q];f=np.column_stack([np.full(len(masks),cal[0,q]),cal[1:,q],drop,drop/capacity,ps[:,0,q],ps[:,1,q],rankdata(drop/capacity)/len(masks)]);assert np.isfinite(f).all();arrays.append(f.astype(np.float32))
            temporary=path.with_suffix('.partial.npz');np.savez_compressed(temporary,features=arrays[0],wrong=arrays[1]);temporary.replace(path);done+=1
            if done%20==0:print('PCMV_NATIVE512',rank,done,round(time.monotonic()-start,1),flush=True)


def ensure(out):
    root=out.parents[1];processes=[];handles=[]
    for rank in range(2):
        h=(out/f'native512-{rank}.log').open('a');handles.append(h);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1')
        processes.append(subprocess.Popen([sys.executable,'-u','-m','experiments.pcmv.native_probe','--root',str(root),'--worker',str(rank)],stdout=h,stderr=subprocess.STDOUT,env=env))
    codes=[p.wait() for p in processes]
    for h in handles:h.close()
    assert codes==[0,0],codes


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int);p.add_argument('--launch',action='store_true');a=p.parse_args();out=a.root/'pcmv-runs/screen-v1'
    if a.launch:ensure(out)
    else:
        assert a.worker is not None
        with (out/f'native512-{a.worker}.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);worker(a.root,out,a.worker,2)
