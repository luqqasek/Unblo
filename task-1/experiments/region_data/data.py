"""50% retained generator, 25% same-content COCO supports, 25% cross-image edits."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import torch
from experiments.native_localizer.generator import EncodedDataset,pair_views,rect_mask,provenance_composite,minority_mask
from experiments.native_localizer.common import rgb_tensor
from wmloc.utils import downsample_mask
BINS=[(.0025,.01),(.01,.05),(.05,.2),(.2,.5)]

def legacy_views(clean,marked,row,rng,pair_kind):
    """Exact old RGB/labels, omitting unused expensive safe-consistency calculation."""
    R=rect_mask(row['rectangle']);kind=row['primary_kind']
    if kind=='clean':support=np.zeros((512,512),bool)
    elif kind=='marked':
        assert R.all();support=R.copy()
    elif row['regime']=='local_crop':support=R.copy()
    else:
        shape=str(rng.choice(['rectangle','strip','components']))
        minor=minority_mask(rng,row['minority_fraction'],shape)
        support=(~minor if row['clean_minority'] else minor)&R
    supports=[support]
    if pair_kind=='clean_twin':supports.append(np.zeros_like(support))
    elif pair_kind=='move':
        assert R.all();offset=(int(rng.integers(64,449)),int(rng.integers(64,449)))
        supports.append(np.roll(support,offset,axis=(0,1)))
    elif pair_kind=='complement':supports.append(R&~support)
    return [provenance_composite(clean,marked,m) for m in supports],[downsample_mask(m) for m in supports],None

class RegionData:
    def __init__(self,root,family,partition,preload=True):
        self.old=EncodedDataset(root/f'encoded/{family}/pilot-{partition}/encoded.jsonl',family)
        self.extra=EncodedDataset(root/f'region-data/{family}/{partition}/extra.jsonl',family)
        if preload:self.old.preload(workers=4);self.extra.preload(workers=2)
        self.rows=self.old.rows+self.extra.rows;self.messages={r['realization_id']:r['message'] for r in self.rows}
        sources={r['source_id']:r for r in map(json.loads,(root/f'sources/coco2017/{partition}.jsonl').read_text().splitlines())}
        assert all(r['partition']==partition for r in self.rows)
        self.full_indices=[i for i,r in enumerate(self.old.rows) if r['rectangle']==[0,0,512,512]]
        self.pool=[];self.bybin={(pol,b):[] for pol in [False,True] for b in range(4)}
        for idx,row in enumerate(self.rows):
            source=sources[row['source_id']];cx,cy,x1,y1=row['source_crop'];scale=512/(x1-cx);R=rect_mask(row['rectangle'])
            accepted=0
            for obj in source.get('objects',[]):
                if not isinstance(obj.get('segmentation'),list):continue
                canvas=Image.new('1',(512,512));draw=ImageDraw.Draw(canvas)
                for poly in obj['segmentation']:
                    if len(poly)>=6:draw.polygon([((poly[k]-cx)*scale,(poly[k+1]-cy)*scale) for k in range(0,len(poly),2)],fill=1)
                mask=np.asarray(canvas,dtype=bool)&R;area=float(mask.mean())
                if not .0025<=area<=.5:continue
                b=next(b for b,(lo,hi) in enumerate(BINS) if lo<=area<=hi)
                item={'index':idx,'object_id':obj.get('id'),'area':area,'mask':np.packbits(mask),'sha256':hashlib.sha256(np.packbits(mask)).hexdigest()}
                n=len(self.pool);self.pool.append(item);self.bybin[(False,b)].append(n)
                if R.all():self.bybin[(True,b)].append(n)
                accepted+=1
                if accepted>=6:break
        assert self.pool
    def load(self,index):return self.old.load(index) if index<len(self.old.rows) else self.extra.load(index-len(self.old.rows))
    def sample(self,rng,novel=False,paired=False):
        if not novel or rng.random()<.5:
            i=int(rng.integers(len(self.old.rows)));clean,marked,row=self.old.load(i)
            kind=None if not paired else (str(rng.choice(['move','clean_twin','complement'])) if row['rectangle']==[0,0,512,512] else 'clean_twin')
            seed=int(rng.integers(2**63));xs,ys,_=legacy_views(clean,marked,row,np.random.default_rng(seed),kind)
            return xs,ys,[dict(realization_id=row['realization_id'],source_id=row['source_id'],background_source_id=row['source_id'],regime=row['regime'],operation='legacy',pair_kind=kind,recipe_seed=seed,primary=j==0,fg_fraction=float(y.mean()),initial_rectangle=row['rectangle'],donor_size=[512,512]) for j,y in enumerate(ys)]
        clean_minority=bool(rng.integers(2));b=int(rng.choice(4,p=[.35,.35,.2,.1]));choices=self.bybin[(clean_minority,b)]
        if not choices:choices=[i for k,v in self.bybin.items() if k[0]==clean_minority for i in v]
        pool_id=int(rng.choice(choices));item=self.pool[pool_id];clean,marked,row=self.load(item['index'])
        mask=np.unpackbits(item['mask']).reshape(512,512).astype(bool)
        # Explicitly simulated removal leaving a thin part; no signal resampling.
        thin=bool(rng.random()<.25)
        if thin:
            mask=mask & ~binary_erosion(mask,iterations=int(rng.integers(2,7)))
            if downsample_mask(mask).sum()==0:mask=np.unpackbits(item['mask']).reshape(512,512).astype(bool);thin=False
        cross=bool(rng.integers(2));background_index=int(rng.choice(self.full_indices)) if clean_minority else int(rng.integers(len(self.old.rows)))
        bg,bgmarked,bgrow=self.old.load(background_index)
        donor_source_id=row['source_id']
        signal_row=bgrow if clean_minority and cross else row
        if clean_minority:
            # A clean donor object overwrites full-image marked base; label excludes it.
            source_clean=np.where(mask[...,None],clean,bg) if cross else clean
            source_marked=np.where(mask[...,None],clean,bgmarked) if cross else provenance_composite(clean,marked,~mask)
            support=~mask;operation='clean_object_addition' if cross else 'same_content_clean_hole'
        else:
            source_clean=np.where(mask[...,None],clean,bg) if cross else clean
            source_marked=np.where(mask[...,None],marked,bg) if cross else provenance_composite(clean,marked,mask)
            support=mask;operation='marked_object_transplant' if cross else 'same_content_object_remnant'
        assert np.array_equal(source_clean[~support],source_marked[~support])
        xs=[source_marked];ys=[downsample_mask(support)]
        if paired:xs.append(source_clean);ys.append(np.zeros((128,128),np.uint8))
        delta=source_marked.astype(np.float32)-source_clean.astype(np.float32);v=delta[support]
        meta=dict(realization_id=signal_row['realization_id'],source_id=signal_row['source_id'],donor_source_id=donor_source_id,background_source_id=bgrow['source_id'] if cross else row['source_id'],regime=row['regime'],operation=operation,object_id=item['object_id'],pool_id=pool_id,object_mask_sha256=item['sha256'],thin_remnant=thin,initial_rectangle=signal_row['rectangle'],donor_size=[512,512],source_crop=row['source_crop'],transform='identity pixel coordinates, hard overwrite; no signal interpolation',support512=int(support.sum()),rms=float(np.sqrt((v*v).mean())),zero_pixel_fraction=float(np.all(v==0,axis=1).mean()),pair_kind='clean_twin' if paired else None)
        return xs,ys,[dict(meta,primary=j==0,fg_fraction=float(y.mean())) for j,y in enumerate(ys)]
    def batch(self,rng,views=32,novel=False):
        xs=[];ys=[];meta=[];pairs=[]
        for g in range(views*3//4):
            a,b,m=self.sample(rng,novel,g<views//4)
            if len(a)==2:pairs.append((len(xs),len(xs)+1))
            xs.extend(rgb_tensor(x) for x in a);ys.extend(torch.tensor(y.copy()).float() for y in b);meta.extend(m)
        return torch.stack(xs),torch.stack(ys),pairs,meta
