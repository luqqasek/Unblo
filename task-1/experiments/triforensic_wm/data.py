"""Unwarped uint8 full images and corresponding 64px clean/marked crops."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from wmloc.data import LocalisationDataset

class Pool:
    def __init__(self,path):
        self.rows=[];self.arrays=[];self.refs=[];self.buckets=[[] for _ in range(5)]
        for shard in sorted(Path(path).glob('shard*')):
            assert (shard/'completed.json').exists(),f'Incomplete shard {shard}'
            rows=[json.loads(s) for s in (shard/'manifest.jsonl').read_text().splitlines()]
            arrays={k:np.load(shard/f'{k}.npy',mmap_mode='r') for k in ('rgb','clean','target','support','noiseprint')}
            sid=len(self.arrays);self.arrays.append(arrays)
            for i,row in enumerate(rows):
                self.buckets[row['bucket']].append(len(self.rows));self.rows.append(row);self.refs.append((sid,i))
        assert len(self.rows)>0
    def get(self,index,key):
        sid,i=self.refs[index];return self.arrays[sid][key][i]
    def full(self,index):
        r=self.rows[index]
        return self.get(index,'rgb'),self.get(index,'target'),np.array([int(c) for c in r['message']],np.float32),self.get(index,'noiseprint')
    def crop(self,index,rng,diagnostic=False):
        m=np.unpackbits(self.get(index,'support')).reshape(512,512).astype(bool)
        yy,xx=np.where(m);assert len(yy)
        # Exact coordinate crop only: no interpolation, padding, flip or rotation.
        if diagnostic or rng.random()<.5:
            r=self.rows[index]
            if r['bucket']==1:x0,y0,x1,y1=r['initial_rectangle'];cx=(x0+x1)//2;cy=(y0+y1)//2
            else:j=int(rng.integers(len(yy)));cx=int(xx[j]);cy=int(yy[j])
            dx=dy=32
        else:
            j=int(rng.integers(len(yy)));cx=int(xx[j]);cy=int(yy[j]);dx=int(rng.integers(8,57));dy=int(rng.integers(8,57))
        x=int(np.clip((cx-dx)//4*4,0,448));y=int(np.clip((cy-dy)//4*4,0,448))
        a=np.array(self.get(index,'clean')[y:y+64,x:x+64]);b=np.array(self.get(index,'rgb')[y:y+64,x:x+64]);support=m[y:y+64,x:x+64]
        target=(support.reshape(16,4,16,4).sum((1,3))>=8).astype(np.float32)
        assert np.array_equal(a[~support],b[~support])
        if not diagnostic:return a,b,target
        # RF29 output centers at input0,4,...60. Padding must never count as interior.
        pure=binary_erosion(support,structure=np.ones((29,29),bool),border_value=0)[::4,::4]
        meta=dict(self.rows[index],crop=[x,y,x+64,y+64],interior_cells=int(pure.sum()))
        return a,b,target,pure,meta

class Engine:
    def __init__(self,root,family):
        self.root=Path(root);self.family=family;self.train=Pool(self.root/f'triforensic-data/{family}/train');self.dev=Pool(self.root/f'triforensic-data/{family}/dev')
        ids=lambda p:{r[k] for r in p.rows for k in ('source_id','background_source_id')}
        assert not ids(self.train)&ids(self.dev),'Source leakage'
        self.ds=LocalisationDataset('data','train',methods=[family]);self.real=[self.ds[i] for i in range(len(self.ds))]
        fold=json.loads(Path('experiments/native_localizer/reconstructed_splits.json').read_text())['families'][family][0]
        lookup={r['image_name']:i for i,r in enumerate(self.real)};self.valid=[lookup[n] for n in fold['validation']];self.training=[i for i in range(len(self.real)) if i not in self.valid]
        assert len(self.training)==113 and len(self.valid)==57
        self.realnp=[np.load(self.root/f"triforensic-data/{family}/real/{r['image_name']}.npy") for r in self.real]
    def real_full(self,i):
        r=self.real[i];return r['image'],r['mask128'],r['message'].astype(np.float32),self.realnp[i]
    @staticmethod
    def stack(items):
        x,y,b,n=zip(*items)
        return {'x':torch.from_numpy(np.stack(x)).permute(0,3,1,2).contiguous(),
                'y':torch.from_numpy(np.stack(y)[:,None]).float(),'bits':torch.from_numpy(np.stack(b)).float(),
                'npmap':torch.from_numpy(np.stack(n)).float()}
    def batch(self,step):
        rng=np.random.default_rng(421007+step);full=[];counts=[0]*5
        # Uniform across5 buckets in expectation, exactly12 synthetic+4 real.
        for j in range(12):
            bucket=(step*12+j)%5;idx=int(rng.choice(self.train.buckets[bucket]));full.append(self.train.full(idx));counts[bucket]+=1
        for i in rng.choice(self.training,4,replace=True):full.append(self.real_full(int(i)))
        result=self.stack(full);pairs=[];targets=[]
        for j in range(64):
            idx=int(rng.choice(self.train.buckets[1+j%2]));a,b,y=self.train.crop(idx,rng)
            pairs.extend([a,b]);targets.extend([np.zeros_like(y),y])
        result['pairs']=torch.from_numpy(np.stack(pairs)).permute(0,3,1,2).contiguous()
        result['pair_y']=torch.from_numpy(np.stack(targets)[:,None]).float();result['counts']=counts
        return result

def device_batch(batch,device='cuda'):
    out={k:v.to(device,non_blocking=True) for k,v in batch.items() if isinstance(v,torch.Tensor)}
    for k in ('x','pairs'):
        if k in out:out[k]=out[k].float()/255
    return out
