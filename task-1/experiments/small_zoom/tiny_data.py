"""Exact crop-first tiny rectangles; source-disjoint, no watermark amplification."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
from experiments.native_localizer.generator import EncodedDataset,rect_mask
from experiments.native_localizer.common import atomic_json,rgb_tensor,sha256
from experiments.region_data.data import legacy_views

def prepare(root,family='pixelseal'):
    wm=load_watermark(family,models_dir='checkpoints',device='cuda')
    source_sets=[]
    for partition,count in [('train',512),('dev',64)]:
        out=root/'tiny-rect-data'/family/partition;out.mkdir(parents=True,exist_ok=True)
        manifest=out/'manifest.json'
        ds=EncodedDataset(root/f'encoded/{family}/pilot-{partition}/encoded.jsonl',family)
        source_sets.append({r['source_id'] for r in ds.rows})
        if manifest.exists():assert len(json.loads(manifest.read_text()))==count;continue
        rng=np.random.default_rng(2701 if partition=='train' else 2702);rows=[]
        for j,index in enumerate(rng.choice(len(ds.rows),count,replace=False)):
            clean,_,old=ds.load(int(index));lo,hi=(.0025,.01) if j%2==0 else (.01,.05)
            area=float(np.exp(rng.uniform(np.log(lo),np.log(hi))))*512**2
            aspect=float(np.exp(rng.uniform(-np.log(4),np.log(4))))
            w=max(8,round(np.sqrt(area*aspect)));h=max(8,round(np.sqrt(area/aspect)))
            assert max(w,h)/min(w,h)<=16
            x=int(rng.integers(513-w));y=int(rng.integers(513-h));patch=clean[y:y+h,x:x+w].copy()
            marked=wm.embed(patch,old['message']);assert marked.dtype==np.uint8 and marked.shape==patch.shape
            path=out/f'{j:04d}.png';Image.fromarray(marked).save(path)
            d=marked.astype(np.float32)-patch.astype(np.float32)
            rows.append({'index':int(index),'source_id':old['source_id'],'rectangle':[x,y,x+w,y+h],'message':old['message'],'patch_path':str(path),'sha256':sha256(path),'rms':float(np.sqrt((d*d).mean())),'zero_pixel_fraction':float(np.all(d==0,2).mean()),'partition':partition,'regime':'new_small_crop_embedded_in_place','embedding_calls':1})
            if (j+1)%128==0:print('PREPARE',partition,j+1,flush=True)
        atomic_json(manifest,rows)
    assert not source_sets[0]&source_sets[1]
    del wm;torch.cuda.empty_cache()

class TinyData:
    def __init__(self,root,partition='train',preload=True):
        self.old=EncodedDataset(root/f'encoded/pixelseal/pilot-{partition}/encoded.jsonl','pixelseal')
        if preload:self.old.preload(workers=4)
        self.rows=json.loads((root/f'tiny-rect-data/pixelseal/{partition}/manifest.json').read_text())
        self.patches=[np.array(Image.open(r['patch_path']).convert('RGB')) for r in self.rows]
    def batch(self,rng,views=32,tiny=False):
        xs=[];ys=[];bs=[];delta=[];regions=[];valid=[];pairs=[]
        count=views//2 if tiny else views
        for g in range(count*3//4):
            clean,marked,row=self.old.load(int(rng.integers(len(self.old.rows))))
            paired=g<count//4
            kind=None if not paired else (str(rng.choice(['move','clean_twin','complement'])) if row['rectangle']==[0,0,512,512] else 'clean_twin')
            images,masks,_=legacy_views(clean,marked,row,rng,kind)
            if paired:pairs.append((len(xs),len(xs)+1))
            for image,y in zip(images,masks):
                xs.append(rgb_tensor(image));ys.append(torch.tensor(y.copy()).float());bs.append(row['message']);delta.append(np.zeros((512,512),np.float32));regions.append(np.zeros((512,512),bool));valid.append(False)
        if tiny:
            for j in range(views*3//8):
                i=int(rng.integers(len(self.rows)));r=self.rows[i];clean,_,_=self.old.load(r['index']);image=clean.copy();x0,y0,x1,y1=r['rectangle'];image[y0:y1,x0:x1]=self.patches[i]
                mask=rect_mask(r['rectangle']);d=((image.astype(np.float32)-clean.astype(np.float32))*np.array([.299,.587,.114],np.float32)).sum(2)
                y=downsample_mask(mask);paired=j<views//8
                if paired:pairs.append((len(xs),len(xs)+1))
                for xx,yy,dd in [(image,y,d)]+([(clean,np.zeros_like(y),np.zeros_like(d))] if paired else []):
                    xs.append(rgb_tensor(xx));ys.append(torch.tensor(yy.copy()).float());bs.append(r['message']);delta.append(dd);regions.append(mask);valid.append(True)
        assert len(xs)==views
        return torch.stack(xs),torch.stack(ys),torch.tensor(bs).float(),pairs,torch.tensor(np.stack(delta)),torch.tensor(np.stack(regions)),torch.tensor(valid)
