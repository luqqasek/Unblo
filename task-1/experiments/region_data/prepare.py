"""Extra exact intermediate embedding rectangles; same train/dev source partitions."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from experiments.native_localizer.generator import EncodedDataset,rect_mask
from experiments.native_localizer.common import atomic_json,sha256
from wmloc.watermarks import load_watermark

def prepare(root,family):
    output=root/'region-data'/family;output.mkdir(parents=True,exist_ok=True)
    wm=None
    for partition,count in [('train',128),('dev',32)]:
        dest=output/partition;dest.mkdir(exist_ok=True);manifest=dest/'extra.jsonl'
        if manifest.exists():
            rows=[json.loads(x) for x in manifest.read_text().splitlines()];assert len(rows)==count;continue
        ds=EncodedDataset(root/f'encoded/{family}/pilot-{partition}/encoded.jsonl',family)
        if wm is None:wm=load_watermark(family,models_dir='checkpoints',device='cuda')
        rng=np.random.default_rng(431 if partition=='train' else 432);rows=[]
        for j,idx in enumerate(rng.choice(len(ds.rows),count,replace=False)):
            clean,_,old=ds.load(int(idx));side=int(rng.choice([128,192,256,384]));x=int(rng.integers(513-side));y=int(rng.integers(513-side))
            marked=clean.copy()
            with torch.inference_mode():patch=wm.embed(clean[y:y+side,x:x+side].copy(),old['message'])
            assert patch.dtype==np.uint8 and patch.shape==(side,side,3)
            marked[y:y+side,x:x+side]=patch;path=dest/f'{j:04d}.png';Image.fromarray(marked).save(path,compress_level=1)
            delta=patch.astype(np.float32)-clean[y:y+side,x:x+side].astype(np.float32)
            row=dict(old,realization_id=old['realization_id']+':intermediate',clean_path=str(ds.base/old['clean_path']),marked_path=str(path),rectangle=[x,y,x+side,y+side],crop_resolution=[side,side],regime='intermediate_crop_remnant',primary_kind='partial',clean_minority=False,minority_fraction=side*side/512**2,marked_sha256=sha256(path),residual_uint8={'rms':float(np.sqrt((delta**2).mean())),'zero_fraction':float((delta==0).mean()),'zero_pixel_fraction':float(np.all(delta==0,2).mean())})
            rows.append(row)
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        atomic_json(dest/'provenance.json',{'family':family,'partition':partition,'embedding_calls':count,'distinct_sources':len({r['source_id'] for r in rows}),'geometry':'square128/192/256/384; crop first, exact embed, paste; no post-embedding resize','source_manifest':sha256(ds.manifest)})
    del wm;torch.cuda.empty_cache()
