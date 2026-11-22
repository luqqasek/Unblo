"""Paired observed RGB patches; exact PNG encodings and source-partition provenance."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from experiments.native_localizer.generator import EncodedDataset,rectangle,rect_mask
from experiments.native_localizer.common import atomic_json
from wmloc.watermarks import load_watermark

def down(m):
    h,w=m.shape;return (m.reshape(h//4,4,w//4,4).sum((1,3))>=8).astype(np.float32)

def prepare(root,family):
    out=root/'local-presence-data'/family;out.mkdir(parents=True,exist_ok=True)
    wm=None;partitions=[]
    for partition,count in [('train',512),('dev',128)]:
        dest=out/partition;dest.mkdir(exist_ok=True)
        if (dest/'manifest.json').exists():
            rows=json.loads((dest/'manifest.json').read_text());partitions.append({r['source_id'] for r in rows});continue
        if wm is None:wm=load_watermark(family,models_dir='checkpoints',device='cuda')
        ds=EncodedDataset(root/f'encoded/{family}/pilot-{partition}/encoded.jsonl',family)
        rng=np.random.default_rng(8701 if partition=='train' else 8702);byid={r['source_id']:i for i,r in enumerate(ds.rows)}
        chosen=rng.choice(list(byid.values()),min(count,len(byid)),replace=False);rows=[]
        for j,index in enumerate(chosen):
            clean,_,old=ds.load(int(index));limit=16 if family=='pixelseal' else 2
            short=int(rng.integers(12,33) if family=='pixelseal' else rng.integers(16,65))
            long=max(short,min(448,int(round(short*np.exp(rng.uniform(0,np.log(limit)))))))
            w,h=(short,long) if rng.integers(2) else (long,short)
            x=int(rng.integers(513-w));y=int(rng.integers(513-h));patch=clean[y:y+h,x:x+w]
            marked=wm.embed(patch,old['message']);Image.fromarray(marked).save(dest/f'{j:04d}.png')
            d=marked.astype(np.float32)-patch.astype(np.float32)
            rows.append({'index':int(index),'source_id':old['source_id'],'rectangle':[x,y,x+w,y+h],'message':old['message'],'file':f'{j:04d}.png','rms':float(np.sqrt((d*d).mean())),'zero_pixel_fraction':float(np.all(d==0,2).mean()),'partition':partition})
            if (j+1)%128==0:print('ENCODED',family,partition,j+1,flush=True)
        atomic_json(dest/'manifest.json',rows);partitions.append({r['source_id'] for r in rows})
    assert not partitions[0]&partitions[1]
    del wm;torch.cuda.empty_cache()

class PatchBank:
    def __init__(self,root,family,partition='train',holes=False,count=2048):
        ds=EncodedDataset(root/f'encoded/{family}/pilot-{partition}/encoded.jsonl',family)
        full=[i for i,r in enumerate(ds.rows) if r['rectangle']==[0,0,512,512]]
        assert full
        folder=root/'local-presence-data'/family/partition
        tiny=json.loads((folder/'manifest.json').read_text()) if not holes else []
        rng=np.random.default_rng(213 if partition=='train' else 214)
        images=[];targets=[];meta=[]
        for j in range(count):
            regime='hole' if holes else ('tiny' if j%3==0 else 'remnant' if j%3==1 else 'hole')
            if regime=='tiny':
                row=tiny[j%len(tiny)];clean,_,_=ds.load(row['index']);marked=clean.copy();x0,y0,x1,y1=row['rectangle'];marked[y0:y1,x0:x1]=np.asarray(Image.open(folder/row['file']))
                support=rect_mask(row['rectangle']);cx=(x0+x1)//2;cy=(y0+y1)//2;sid=row['source_id']
            else:
                index=int(rng.choice(full));clean,marked,row=ds.load(index);sid=row['source_id']
                cx=int(rng.integers(48,465));cy=int(rng.integers(48,465))
                short=int(rng.integers(12,57));long=int(rng.integers(short,97));w,h=(short,long) if rng.integers(2) else (long,short)
                x0=max(0,cx-w//2);y0=max(0,cy-h//2);x1=min(512,x0+w);y1=min(512,y0+h)
                support=rect_mask([x0,y0,x1,y1])
                if j%4==0:
                    yy,xx=np.ogrid[:512,:512];support &= ((xx-cx)/max(1,w/2))**2+((yy-cy)/max(1,h/2))**2<=1
                support=~support if regime=='hole' else support
            # Grid-aligned crop, no resampling. Identical content/paste boundaries in twins.
            x=int(np.clip((cx-32+int(rng.integers(-12,13)))//4*4,0,448));y=int(np.clip((cy-32+int(rng.integers(-12,13)))//4*4,0,448))
            c=clean[y:y+64,x:x+64];w=marked[y:y+64,x:x+64];m=support[y:y+64,x:x+64]
            if regime=='hole':
                a=w;b=np.where(m[...,None],w,c);ma=np.ones((64,64),bool);mb=m
            else:
                a=c;b=np.where(m[...,None],w,c);ma=np.zeros((64,64),bool);mb=m
            images.append(np.stack([a,b]));targets.append(np.stack([down(ma),down(mb)]));meta.append({'source_id':sid,'regime':regime,'crop':[x,y,x+64,y+64],'area':float(m.mean())})
        self.images=torch.tensor(np.stack(images)).permute(0,1,4,2,3).contiguous()
        self.targets=torch.tensor(np.stack(targets));self.meta=meta
    def batch(self,rng,pairs=64):
        idx=rng.integers(len(self.images),size=pairs)
        return self.images[idx].flatten(0,1).cuda().float()/255,self.targets[idx].flatten(0,1).cuda()

def full_examples(root,family,holes=False,count=128):
    ds=EncodedDataset(root/f'encoded/{family}/pilot-train/encoded.jsonl',family)
    full=[i for i,r in enumerate(ds.rows) if r['rectangle']==[0,0,512,512]]
    folder=root/'local-presence-data'/family/'train'
    tiny=json.loads((folder/'manifest.json').read_text()) if not holes else []
    rng=np.random.default_rng(8173)
    for j in range(count):
        if not holes and j%3==0:
            row=tiny[j%len(tiny)];clean,_,_=ds.load(row['index']);marked=clean.copy();x0,y0,x1,y1=row['rectangle'];marked[y0:y1,x0:x1]=np.array(Image.open(folder/row['file']));support=rect_mask(row['rectangle'])
        else:
            clean,marked,row=ds.load(int(rng.choice(full)))
            fraction=float(np.exp(rng.uniform(np.log(.0025),np.log(.05))))
            support=rect_mask(rectangle(rng,fraction,16))
            if holes or j%3==2:support=~support
        image=np.where(support[...,None],marked,clean).astype(np.uint8)
        yield image,down(support),row['message']
