"""CPU-only exact clean/marked residual audit, in an existing allocation."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 for family in ['pixelseal','trustmark']:
  output=[]
  for manifest in [a.root/f'encoded/{family}/pilot-dev/encoded.jsonl',a.root/f'region-data/{family}/dev/extra.jsonl']:
   for row in map(json.loads,manifest.read_text().splitlines()):
    clean=np.asarray(Image.open(manifest.parent/row['clean_path']).convert('RGB'));marked=np.asarray(Image.open(manifest.parent/row['marked_path']).convert('RGB'))
    x0,y0,x1,y1=row['rectangle'];inside=np.zeros((512,512),bool);inside[y0:y1,x0:x1]=True
    assert np.array_equal(clean[~inside],marked[~inside]),'Changes outside declared embedding rectangle'
    c=clean[y0:y1,x0:x1].astype(np.float32);delta=marked[y0:y1,x0:x1].astype(np.float32)-c
    gradients=[float(abs(np.diff(c,axis=ax)).mean()) for ax in [0,1] if c.shape[ax]>1]
    output.append({'realization_id':row['realization_id'],'regime':row['regime'],'rectangle':row['rectangle'],'initial_area':float(inside.mean()),'rms':float(np.sqrt((delta**2).mean())),'zero_channels':float((delta==0).mean()),'zero_pixels':float(np.all(delta==0,2).mean()),'clean_gradient':float(np.mean(gradients)) if gradients else None})
  out=a.root/'region-runs'/family/'encoding_pixels_dev.json';out.write_text(json.dumps(output,indent=2));print(family,len(output),'exact residuals audited',flush=True)
if __name__=='__main__':main()
