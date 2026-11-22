"""Descriptive local texture and boundary statistics, no distribution fitting."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import sobel,binary_erosion
import torch
from wmloc.data import LocalisationDataset
from .data import RegionData

def measure(x,y):
    gray=x.astype(np.float32).mean(2)
    g=np.hypot(sobel(gray,0),sobel(gray,1))/8
    g=g.reshape(128,4,128,4).mean((1,3));m=y.astype(bool);m=m if m.mean()<=.5 else ~m
    edge=m&~binary_erosion(m);edge[[0,-1],:]=False;edge[:,[0,-1]]=False
    inside=m&~edge
    mean=lambda mask:float(g[mask].mean()) if mask.any() else None
    return dict(minority_texture_gradient=mean(m),boundary_texture_gradient=mean(edge),interior_texture_gradient=mean(inside),global_texture_gradient=float(g.mean()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 for f in ['pixelseal','trustmark']:
  data=RegionData(a.root,f,'dev',preload=False);rows=[]
  for variant in ['control','objects']:
   rng=np.random.default_rng(991)
   for i in range(128):
    xs,ys,meta=data.sample(rng,variant=='objects',True);rows.append(dict(variant=variant,**meta[0],**measure(xs[0],ys[0])))
  for item in LocalisationDataset(a.root/'task-1/data','train',methods=[f]):rows.append(dict(variant='real',image_name=item['image_name'],fg_fraction=float(item['mask128'].mean()),**measure(item['image'],item['mask128'])))
  (a.root/'region-runs'/f/'texture_audit.json').write_text(json.dumps(rows,indent=2));print(f,len(rows),'texture descriptions complete',flush=True)
if __name__=='__main__':main()
