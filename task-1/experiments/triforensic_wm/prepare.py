"""Exact-embedder uint8 pool, provenance twins, source-disjoint cached Noiseprint++."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scipy.ndimage import binary_erosion
from experiments.native_localizer.generator import rect_mask,rectangle
from experiments.native_localizer.common import atomic_json,rgb_tensor
from experiments.region_data.data import RegionData
from experiments.local_presence.data import down
from .noiseprint import Noiseprint
from wmloc.watermarks import load_watermark
from wmloc.data import LocalisationDataset

def irregular(rng,area):
    yy,xx=np.ogrid[:512,:512];distance=np.full((512,512),np.inf);n=int(rng.integers(1,4))
    for _ in range(n):
        x,y=rng.uniform(0,512,2);ratio=np.exp(rng.uniform(-1.5,1.5));d=((xx-x)/ratio)**2+((yy-y)*ratio)**2;distance=np.minimum(distance,d)
    k=max(1,min(512*512-1,round(area*512*512)));idx=np.argpartition(distance.ravel(),k-1)[:k];m=np.zeros(512*512,bool);m[idx]=1;return m.reshape(512,512)

def shift_paste(base,donor,mask,rng):
    ys,xs=np.where(mask);h,w=ys.max()-ys.min()+1,xs.max()-xs.min()+1;x=int(rng.integers(513-w));y=int(rng.integers(513-h));patch=donor[ys.min():ys.max()+1,xs.min():xs.max()+1];cut=mask[ys.min():ys.max()+1,xs.min():xs.max()+1];out=base.copy();out[y:y+h,x:x+w][cut]=patch[cut];m=np.zeros_like(mask);m[y:y+h,x:x+w]=cut;return out,m,[x-int(xs.min()),y-int(ys.min())]

class Maker:
    def __init__(self,root,family,partition):
        self.data=RegionData(root,family,partition);self.family=family;self.partition=partition;self.wm=load_watermark(family,models_dir='checkpoints',device='cuda');self.calls=0
        self.small_objects=[p for p in self.data.pool if .003<=p['area']<=.05]
        # Label-independent low-texture ranking; PS uses exact official JND.
        scores=[]
        with torch.no_grad():
            for begin in range(0,len(self.data.old.rows),16):
                items=[self.data.old.load(i)[0] for i in range(begin,min(begin+16,len(self.data.old.rows)))];x=torch.stack([rgb_tensor(it) for it in items]).cuda()
                if family=='pixelseal':v=self.wm.model.attenuation.heatmaps(x).mean((1,2,3))
                else:v=(x[:,:,1:]-x[:,:,:-1]).abs().mean((1,2,3))+(x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean((1,2,3))
                scores.extend(v.cpu().tolist())
        self.low=np.argsort(scores)[:max(1,len(scores)//4)]
    def encode(self,cover,bits):
        out=self.wm.embed(cover.copy(),bits);self.calls+=1;assert out.dtype==np.uint8 and out.shape==cover.shape;return out
    def sample(self,index,bucket):
        rng=np.random.default_rng((81277 if self.partition=='train' else 91277)+index);low=bucket==3 and rng.random()<.3;obj=None
        if bucket==2 and self.small_objects and rng.random()<.5:
            obj=self.small_objects[int(rng.integers(len(self.small_objects)))];clean,_,row=self.data.load(obj['index'])
        else:
            idx=int(rng.choice(self.low)) if low else int(rng.integers(len(self.data.old.rows)));clean,_,row=self.data.old.load(idx)
        bits=rng.integers(0,2,self.wm.nbits,dtype=np.int64);R=[0,0,512,512];transform=[0,0];bgid=row['source_id'];kind='';support=np.ones((512,512),bool);twin=clean.copy();before=self.calls
        if bucket==1:
            limit=16 if self.family=='pixelseal' else 2
            for _ in range(1000):
                short=int(rng.integers(12,33)) if self.family=='pixelseal' else int(rng.integers(16,97));long=min(512,round(short*np.exp(rng.uniform(0,np.log(limit)))));w,h=(short,long) if rng.integers(2) else (long,short)
                if .002<=w*h/512**2<=.05:break
            else:raise RuntimeError('Invalid tiny geometry')
            x=int(rng.integers(513-w));y=int(rng.integers(513-h));R=[x,y,x+w,y+h];marked=clean.copy();marked[y:y+h,x:x+w]=self.encode(clean[y:y+h,x:x+w],bits);support=rect_mask(R);kind='small_crop_embedding'
        elif bucket==4 and index%3==0:
            marked=clean.copy();support[:]=False;kind='fully_clean'
        else:
            base=self.encode(clean,bits)
            if bucket==0:
                area=float(rng.uniform(.05,.85));support=rect_mask(rectangle(rng,area,16 if self.family=='pixelseal' else 2)) if index%2==0 else irregular(rng,area);kind='standard_full_source_edit';marked=np.where(support[...,None],base,clean)
            elif bucket==2:
                support=np.unpackbits(obj['mask']).reshape(512,512).astype(bool) if obj is not None else irregular(rng,float(np.exp(rng.uniform(np.log(.003),np.log(.05)))))
                if index%2==0:
                    bg,_,brow=self.data.old.load(int(rng.integers(len(self.data.old.rows))));bgid=brow['source_id']
                    while bgid==row['source_id']:bg,_,brow=self.data.old.load(int(rng.integers(len(self.data.old.rows))));bgid=brow['source_id']
                    state=rng.bit_generator.state;marked,m,transform=shift_paste(bg,base,support,rng);rng.bit_generator.state=state;twin,_,_=shift_paste(bg,clean,support,rng);support=m;kind='translated_large_source_fragment'
                else:marked=np.where(support[...,None],base,clean);kind='in_place_large_source_fragment'
            elif bucket==3 and not low:
                area=float(np.exp(rng.uniform(np.log(.003),np.log(.05))));hole=irregular(rng,area) if index%2 else rect_mask(rectangle(rng,area,16));support=~hole;mode=index%3
                if mode==0:edit=clean;kind='same_content_clean_hole'
                elif mode==1:edit,_,brow=self.data.old.load(int(rng.integers(len(self.data.old.rows))));bgid=brow['source_id'];kind='clean_natural_crop_hole'
                else:edit=np.broadcast_to(clean.mean((0,1)).round().astype(np.uint8),(512,512,3));kind='flat_clean_hole'
                marked=np.where(hole[...,None],edit,base);twin=np.where(hole[...,None],edit,clean)
            elif bucket==3:marked=base;kind='intact_low_observability'
            elif index%3==1:marked=base;kind='fully_marked'
            else:
                support=irregular(rng,float(rng.uniform(.003,.2)));marked=np.where(support[...,None],base,clean);kind='disconnected_fragments'
        assert np.array_equal(marked[~support],twin[~support]);y=down(support);delta=marked.astype(np.int16)-twin.astype(np.int16);vals=delta[support]
        meta={'index':index,'bucket':bucket,'operation':kind,'source_id':row['source_id'],'background_source_id':bgid,'partition':self.partition,'message':''.join(map(str,bits.tolist())),'initial_rectangle':R,'source_crop':row['source_crop'],'donor_size':[512,512],'translation':transform,'foreground_fraction':float(y.mean()),'embed_calls':self.calls-before,'rms_uint8':float(np.sqrt(np.mean(vals.astype(float)**2))) if vals.size else 0.,'unchanged_pixel_fraction_in_support':float(np.all(vals==0,axis=1).mean()) if vals.size else None,'support_from':'pixel provenance, not RGB differences'}
        return marked.astype(np.uint8),twin.astype(np.uint8),support,y,meta

def prepare(root,family,part,partition,count):
    out=root/f'triforensic-data/{family}/{partition}/shard{part}';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return
    started=time.monotonic();maker=Maker(root,family,partition);noise=Noiseprint().cuda();n=len(range(part,count,4));paths={}
    for name,shape,dtype in [('rgb',(n,512,512,3),np.uint8),('clean',(n,512,512,3),np.uint8),('support',(n,32768),np.uint8),('target',(n,128,128),np.uint8),('noiseprint',(n,1,128,128),np.float32)]:paths[name]=np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)
    records=[];pending=[];pending_ids=[]
    for local,index in enumerate(range(part,count,4)):
        # Exact pool proportions20/20/20/25/15; training sampler is uniform by bucket.
        q=index%20;bucket=0 if q<4 else 1 if q<8 else 2 if q<12 else 3 if q<17 else 4
        x,c,m,y,row=maker.sample(index,bucket);paths['rgb'][local]=x;paths['clean'][local]=c;paths['support'][local]=np.packbits(m);paths['target'][local]=y.astype(np.uint8);records.append(row);pending.append(rgb_tensor(x));pending_ids.append(local)
        if len(pending)==16 or local+1==n:
            with torch.no_grad():npmap=noise(torch.stack(pending).cuda()).cpu().numpy()
            paths['noiseprint'][pending_ids]=npmap;pending=[];pending_ids=[]
        if (local+1)%128==0:print('PREPARE',family,partition,part,local+1,n,'seconds',time.monotonic()-started,flush=True)
    for a in paths.values():a.flush()
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));atomic_json(out/'completed.json',{'n':n,'embed_calls':maker.calls,'seconds':time.monotonic()-started,'source_ids':sorted({r['source_id'] for r in records}),'bucket_counts':{str(b):sum(r['bucket']==b for r in records) for b in range(5)}})
    del maker,noise,paths;torch.cuda.empty_cache()
    if partition=='dev':
        ds=LocalisationDataset('data','train',methods=[family]);model=Noiseprint().cuda();real=root/f'triforensic-data/{family}/real';real.mkdir(exist_ok=True)
        with torch.no_grad():
            for i in range(part,len(ds),4):
                it=ds[i];np.save(real/f"{it['image_name']}.npy",model(rgb_tensor(it['image'])[None].cuda())[0].cpu().numpy())
        print('REAL_NOISEPRINT_CACHED',family,part,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--part',type=int,required=True);p.add_argument('--count',type=int,default=10000);a=p.parse_args();torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    prepare(a.root,a.family,a.part,'train',a.count);prepare(a.root,a.family,a.part,'dev',640)
if __name__=='__main__':main()
