"""Prepare all official train records and the complete existing synthetic master pools."""
import argparse, hashlib, importlib.util, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'
FAMILIES=('pixelseal','trustmark','stegastamp')

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def mmap(path,shape,dtype):
    if path.exists():
        a=np.load(path,mmap_mode='r+');assert a.shape==shape and a.dtype==np.dtype(dtype),(path,a.shape,shape);return a
    return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)

def geometry(mask):
    # Target construction only. Neither the mask nor geometry is an inference input.
    from wmloc.utils import downsample_mask
    y=downsample_mask(mask).astype(bool)
    edge=ndimage.maximum_filter(mask,size=3)!=ndimage.minimum_filter(mask,size=3)
    edge=edge.reshape(128,4,128,4).max((1,3)).astype(np.float32)
    distance=np.clip((ndimage.distance_transform_edt(y)-ndimage.distance_transform_edt(~y))/16,-1,1)
    centers=[]
    for binary in (y,~y):
        labels,n=ndimage.label(binary,np.ones((3,3)))
        center=np.zeros((128,128),np.float32)
        for k,s in enumerate(ndimage.find_objects(labels),1):
            if s is None:continue
            region=labels[s]==k
            cy,cx=ndimage.center_of_mass(region)
            cy=int(np.clip(round(cy+s[0].start),0,127));cx=int(np.clip(round(cx+s[1].start),0,127))
            center[cy,cx]=1
        center=ndimage.gaussian_filter(center,1)
        if center.max()>0:center/=center.max()
        centers.append(center)
    return np.stack([edge,distance,*centers]).astype(np.float16)

class Evidence:
    def __init__(self,family,source):
        from wmloc.watermarks import load_watermark
        self.family=family;self.source=source
        self.wm=load_watermark(family,ROOT/'checkpoints','cpu' if family=='stegastamp' else 'cuda')
        self.collected={};self.hooks=[]
        if family=='stegastamp':
            import onnxruntime as ort
            opts=ort.SessionOptions();opts.intra_op_num_threads=4
            self.wm.sess=ort.InferenceSession(str(ROOT/'experiments/synthetic_pretrain_2k/results/cache/features.onnx'),opts,providers=['CPUExecutionProvider'])
        else:
            layers=list(self.wm.det.convnext.stages) if family=='pixelseal' else [getattr(self.wm.net,f'layer{i}') for i in range(1,5)]
            for k,layer in enumerate(layers):self.hooks.append(layer.register_forward_hook(lambda m,i,o,key=k:self.collected.__setitem__(key,o.detach())))
    @torch.inference_mode()
    def extract(self,image,message,known_features=None,bit_size=16):
        x=torch.from_numpy(image.copy()).permute(2,0,1).float()[None]/255
        msg=torch.tensor(message,dtype=torch.float32)
        if self.family=='stegastamp':
            names=self.source.FEATURE_NAMES+['gen_stn/Reshape:0',self.wm.LOGITS]
            values=self.wm.sess.run(names,{'image':self.wm._prep(x),'secret':np.zeros((1,100),np.float32)})
            fs=[]
            for k,v in enumerate(values[:4]):
                f=torch.from_numpy(v);f=F.adaptive_avg_pool2d(f,(min(32,f.shape[-2]),min(32,f.shape[-1])))
                f=F.grid_sample(f,self.source.inverse_feature_grid(values[-2][0],k,f.shape[-1]),mode='bilinear',align_corners=False,padding_mode='border')
                fs.append(f[0].numpy())
            signed=torch.from_numpy(values[-1][0])*(2*msg-1)
            bits=torch.tanh(signed/4)[:,None,None]
            scalar=torch.stack([torch.tanh(signed.mean()/4),signed.sigmoid().mean(),signed.mean()/signed.square().mean().sqrt().clamp_min(1e-6)])[:,None,None]
        else:
            x=x.cuda();self.wm.logits(x)
            fs=[F.adaptive_avg_pool2d(self.collected[k],(min(32,self.collected[k].shape[-2]),min(32,self.collected[k].shape[-1])))[0].cpu().numpy() for k in range(4)]
            def projected(last):
                if self.family=='pixelseal':
                    v=self.wm.det.pixel_decoder.output_upscaling(last);fc=self.wm.det.pixel_decoder.linear
                    return F.conv2d(v,fc.weight[:,:,None,None],fc.bias)[:,1:]
                fc=self.wm.net.fc;return F.conv2d(last,fc.weight[:,:,None,None],fc.bias)
            native=projected(self.collected[3])
            if self.family=='pixelseal':self.wm.det(x)
            else:self.wm.net(x*2-1)
            high=projected(self.collected[3]);sgn=(2*msg.cuda()-1)[None,:,None,None]
            if bit_size is None:bit_size=max(native.shape[-1],high.shape[-1])
            bb=[];ss=[]
            for z in (native,high):
                signed=z*sgn
                bb.append(F.interpolate(torch.tanh(signed/4),(bit_size,bit_size),mode='bilinear',align_corners=False))
                score=torch.cat([torch.tanh(signed.mean(1,keepdim=True)/4),signed.sigmoid().mean(1,keepdim=True),signed.mean(1,keepdim=True)/signed.square().mean(1,keepdim=True).sqrt().clamp_min(1e-6)],1)
                ss.append(F.interpolate(score,(16,16),mode='bilinear',align_corners=False))
            bits=torch.cat(bb,1)[0].cpu();scalar=torch.cat(ss,1)[0].cpu()
        return fs,bits.numpy().astype(np.float16),scalar.numpy().astype(np.float16)

def prepare(family):
    os.chdir(ROOT);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert '3070' in torch.cuda.get_device_name(0),torch.cuda.get_device_name(0)
    folder=OUT/'data'/family;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    source=module('allinfo_source',ROOT/'experiments/synthetic_pretrain_2k/prepare.py')
    old=ROOT/'experiments/synthetic_scaling/results/data'/family
    rows=json.loads((old/'manifest.json').read_text(encoding='utf-8'));assert len(rows)==4200
    metadata=json.loads((ROOT/'data/train/metadata.json').read_text(encoding='utf-8'))
    real=[r for r in metadata if r['method']==family];assert len(real)==170
    train_sources={r['coco_id'] for r in rows if r['split']=='train'};dev_sources={r['coco_id'] for r in rows if r['split']!='train'}
    assert len(train_sources)==4000 and len(dev_sources)==200 and not train_sources&dev_sources
    with zipfile.ZipFile(ROOT/'experiments/synthetic_pretrain_2k/results/sources/annotations_trainval2017.zip') as z:ann=json.loads(z.read('annotations/instances_val2017.json'))
    byid={}
    for a in ann['annotations']:byid.setdefault(a['image_id'],[]).append(a)
    official=np.load(ROOT/'data/train/masks_128.npz');official=dict(zip(official['names'].astype(str),official['masks']))
    previous=np.load(old/'targets.npy',mmap_mode='r')
    arrays={};n=4370;started=time.monotonic();evidence=Evidence(family,source)
    records=[];mismatches=0
    archive=zipfile.ZipFile(ROOT/'experiments/synthetic_pretrain_2k/results/sources/val2017.zip')
    progress=folder/'progress.json';done=json.loads(progress.read_text())['completed'] if progress.exists() else 0
    old_records=json.loads((folder/'records.json').read_text()) if (folder/'records.json').exists() else []
    if done:assert len(old_records)==done
    records=old_records
    for i in range(n):
        if i<done:continue
        if i<4200:
            r=rows[i];rng=np.random.default_rng(20261121+i)
            clean,region,shape,crop=source.cover_and_mask(archive.read('val2017/'+r['source_filename']),byid.get(r['coco_id'],[]),rng,['rectangle','ellipse','object'][i%3])
            assert hashlib.sha256(clean.tobytes()).hexdigest()==r['cover_sha256'],('cover',i)
            bits=rng.integers(0,2,evidence.wm.nbits,dtype=np.uint8);assert ''.join(map(str,bits.tolist()))==r['message']
            mode=r['mode'];mask=np.zeros((512,512),bool) if mode=='clean' else np.ones((512,512),bool) if mode=='full' else ~region if mode=='restore' else region
            assert np.array_equal(source.downsample_mask(mask),previous[i]),('mask',family,i)
            cached=(ROOT/'experiments/synthetic_pretrain_2k/results/images'/f"{r['image_id']}.png") if family=='stegastamp' and i<2200 else None
            if cached is not None and cached.exists():image=np.array(Image.open(cached).convert('RGB'))
            elif mode=='clean':image=clean.copy()
            elif mode in ('full','restore'):
                marked=evidence.wm.embed(clean,bits);image=np.where(mask[:,:,None],marked,clean)
            else:
                yy,xx=np.where(region);x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
                if r.get('embedding_bbox') is not None:x0,y0,x1,y1=r['embedding_bbox']
                marked=clean.copy();marked[y0:y1,x0:x1]=evidence.wm.embed(clean[y0:y1,x0:x1],bits);image=np.where(mask[:,:,None],marked,clean)
            assert np.array_equal(image[~mask],clean[~mask])
            mismatch=hashlib.sha256(image.tobytes()).hexdigest()!=r['image_sha256'];mismatches+=int(mismatch)
            record={'name':r['image_id'],'source_id':str(r['coco_id']),'partition':r['split'],'synthetic':True,'full_watermark':mode in ('full','restore'),'historical_rgb_mismatch':mismatch,'image_sha256':hashlib.sha256(image.tobytes()).hexdigest()}
        else:
            r=real[i-4200];bits=np.array(list(map(int,r['message'])),np.uint8)
            image=np.array(Image.open(ROOT/'data/train/images'/f"{r['image_name']}.png").convert('RGB'))
            mask=np.array(Image.open(ROOT/'data/train/masks'/f"{r['image_name']}.png"))>127;clean=np.zeros_like(image)
            assert np.array_equal(source.downsample_mask(mask),official[r['image_name']])
            record={'name':r['image_name'],'source_id':r['image_name'],'partition':'real','synthetic':False,'full_watermark':bool(r['full_watermark']),'image_sha256':hashlib.sha256(image.tobytes()).hexdigest()}
        fs,bit_maps,scalar=evidence.extract(image,bits)
        energy=np.abs(image.astype(np.float32)-clean.astype(np.float32)).mean(2).reshape(128,4,128,4).mean((1,3)) if i<4200 else np.zeros((128,128),np.float32)
        values={'rgb':image.transpose(2,0,1),'mask512':mask.astype(np.uint8),'mask128':source.downsample_mask(mask).astype(np.uint8),'geometry':geometry(mask),'energy':energy.astype(np.float16),'message':bits,'bits':bit_maps,'scalar':scalar,'clean':clean.transpose(2,0,1)}
        values.update({f'feature{k}':v.astype(np.float16) for k,v in enumerate(fs)})
        for key,v in values.items():
            if key not in arrays:arrays[key]=mmap(folder/f'{key}.npy',(n,*v.shape),v.dtype)
            arrays[key][i]=v
        records.append(record)
        if (i+1)%25==0 or i+1==n:
            for a in arrays.values():a.flush()
            save(folder/'records.json',records);save(progress,{'completed':i+1,'total':n,'seconds':time.monotonic()-started})
            elapsed=time.monotonic()-started;print('PREPARE',family,i+1,n,'seconds',round(elapsed,1),'eta',round(elapsed/(i+1-done)*(n-i-1),1),flush=True)
    assert len(records)==4370
    save(folder/'complete.json',{'family':family,'n':n,'real':170,'synthetic_train':4000,'synthetic_dev':200,'historical_rgb_mismatches':sum(r.get('historical_rgb_mismatch',False) for r in records),'features':'recomputed on exact current pixels, original frozen decoders','seconds':time.monotonic()-started,'source_manifest_sha256':sha(old/'manifest.json'),'gpu':torch.cuda.get_device_name(0),'peak_gpu_bytes':torch.cuda.max_memory_allocated()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=FAMILIES,required=True);a=p.parse_args();prepare(a.family)
