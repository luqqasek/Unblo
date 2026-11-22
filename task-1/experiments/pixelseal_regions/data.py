"""Source-grouped exact pairs, including GT-clean real TRAIN regions only."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from experiments.native_localizer.generator import EncodedDataset
from experiments.native_localizer.common import atomic_json,sha256
from .geometry import down,tight


def real_split(root):
    old=json.loads((root/'triforensic-runs/pixelseal/unified-v1/split.json').read_text())
    ds=LocalisationDataset('data','train',methods=['pixelseal']);valid=set(old['validation']);training=sorted(r['image_name'] for r in ds if r['image_name'] not in valid)
    rng=np.random.default_rng(8831);rng.shuffle(training)
    return {'train':training[:90],'calibration':training[90:],'validation':sorted(valid),'seed':8831}


def make_plan(root,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():return
    split=real_split(root);atomic_json(out/'split.json',split)
    tasks=[]
    for role,count in [('train',4),('calibration',3)]:
        for name in split[role]:
            for v in range(count):tasks.append({'kind':'real_clean','role':role,'source':name,'variant':v})
    for role,partition,count in [('train','train',256),('calibration','dev',64),('audit','dev',64)]:
        ds=EncodedDataset(root/f'encoded/pixelseal/pilot-{partition}/encoded.jsonl','pixelseal');unique={r['source_id']:i for i,r in enumerate(ds.rows)};ids=sorted(unique)
        if role=='audit':ids=ids[64:128]
        else:ids=ids[:count]
        for sid in ids:tasks.append({'kind':'public','role':role,'source':sid,'partition':partition,'index':unique[sid],'variant':0})
    for i,t in enumerate(tasks):t['id']=f'{i:05d}';t['seed']=15091+i
    sets={role:{t['source'] for t in tasks if t['role']==role} for role in ('train','calibration','audit')}
    assert not sets['train']&sets['calibration'] and not sets['calibration']&sets['audit']
    atomic_json(out/'plan.json',{'split':split,'tasks':tasks,'source_sets':{k:sorted(v) for k,v in sets.items()},'original_embedder':'PixelSeal official wrapper, unchanged strength, FP32 and uint8 rounding','post_embedding_augmentation':False})


def prep(root,out,rank,world):
    plan=json.loads((out/'plan.json').read_text());(out/'pairs').mkdir(exist_ok=True)
    real={r['image_name']:r for r in LocalisationDataset('data','train',methods=['pixelseal'])}
    public={p:EncodedDataset(root/f'encoded/pixelseal/pilot-{p}/encoded.jsonl','pixelseal') for p in ('train','dev')}
    wm=load_watermark('pixelseal',models_dir='checkpoints',device=f'cuda:{rank}');start=time.monotonic();calls=0;rows=[]
    for task in plan['tasks'][rank::world]:
        path=out/'pairs'/f"{task['id']}.npz";meta=path.with_suffix('.json')
        if meta.exists():rows.append(json.loads(meta.read_text()));continue
        rng=np.random.default_rng(task['seed'])
        if task['kind']=='real_clean':
            item=real[task['source']];image=item['image'];bad=item['mask'].astype(np.int64);ii=np.pad(bad.cumsum(0).cumsum(1),((1,0),(1,0)))
        else:
            source=public[task['partition']];image=np.asarray(Image.open(source.base/source.rows[task['index']]['clean_path']).convert('RGB'));ii=None
        found=False
        for attempt in range(100):
            short=int(rng.integers(12,49));long=int(np.clip(round(short*np.exp(rng.uniform(0,np.log(16)))),short,384));w,h=(short,long) if rng.integers(2) else (long,short)
            if not 512<=w*h<=13107:continue
            H=(h+39)//4*4;W=(w+39)//4*4
            if ii is None:x=int(rng.integers(513-W));y=int(rng.integers(513-H));found=True
            else:
                counts=ii[H:,W:]-ii[:-H,W:]-ii[H:,:-W]+ii[:-H,:-W];yy,xx=np.where(counts==0)
                if len(xx):j=int(rng.integers(len(xx)));x=int(xx[j]);y=int(yy[j]);found=True
            if found:break
        if not found:
            record=dict(task,skipped='No sufficiently large entirely clean native crop');atomic_json(meta,record);rows.append(record);continue
        clean=np.array(image[y:y+H,x:x+W]);marked=clean.copy();box=[16,16,16+w,16+h];bits=rng.integers(2,size=256).astype(np.uint8);patch=wm.embed(clean[16:16+h,16:16+w],bits);calls+=1;marked[16:16+h,16:16+w]=patch
        mask=np.zeros((H,W),np.uint8);mask[16:16+h,16:16+w]=1
        assert np.array_equal(marked[mask==0],clean[mask==0]);assert clean.dtype==marked.dtype==np.uint8
        if task['kind']=='real_clean':assert not real[task['source']]['mask'][y:y+H,x:x+W].any()
        delta=marked.astype(np.float32)-clean.astype(np.float32);d=delta[mask.astype(bool)]
        record=dict(task,skipped=None,cover_crop=[x,y,x+W,y+H],embedding_rectangle=box,original_crop_size=[h,w],message=''.join(map(str,bits.tolist())),rms=float(np.sqrt((d*d).mean())),zero_fraction=float((d==0).all(1).mean()),clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),marked_sha256=hashlib.sha256(marked.tobytes()).hexdigest())
        np.savez_compressed(path,clean=clean,marked=marked,mask=down(mask),box=np.array(box),bits=bits);atomic_json(meta,record);rows.append(record)
        if calls%64==0:print('PAIRS',rank,calls,round(time.monotonic()-start,1),flush=True)
    atomic_json(out/f'prepare-{rank}.json',{'new_embedding_calls':calls,'seconds':time.monotonic()-start,'planned':len(rows),'successful':sum(r['skipped'] is None for r in rows),'sources':sorted({r['source'] for r in rows if r['skipped'] is None})})
    del wm;torch.cuda.empty_cache()


class Bank:
    def __init__(self,out):
        self.split=json.loads((out/'split.json').read_text());self.real={r['image_name']:r for r in LocalisationDataset('data','train',methods=['pixelseal'])};self.rows={k:[] for k in ('train','calibration','audit')}
        for p in sorted((out/'pairs').glob('*.json')):
            row=json.loads(p.read_text())
            if row['skipped'] is None:
                with np.load(p.with_suffix('.npz')) as z:row.update({k:z[k] for k in z.files})
                self.rows[row['role']].append(row)
        assert all(self.rows.values())
    def batch(self,rng):
        rows=self.rows['train'];realclean=[r for r in rows if r['kind']=='real_clean'];public=[r for r in rows if r['kind']=='public'];items=[]
        for pool in (realclean or public,public):
            r=pool[int(rng.integers(len(pool)))];h,w=r['mask'].shape;box=tight(r['mask'])
            for marked in (False,True):items.append((r['marked'] if marked else r['clean'],r['mask'] if marked else np.zeros_like(r['mask']),box))
        for name in rng.choice(self.split['train'],2,replace=False):
            r=self.real[name];items.append((r['image'],r['mask128'],tight(r['mask128'])))
        return items


def pack(items,device):
    H=max(x.shape[0] for x,_,_ in items);W=max(x.shape[1] for x,_,_ in items);arr=np.zeros((len(items),H,W,3),np.uint8)
    for i,(x,_,_) in enumerate(items):arr[i,:x.shape[0],:x.shape[1]]=x
    return torch.from_numpy(arr).permute(0,3,1,2).to(device).float()/255
