"""Render predeclared coverage examples; GT used only to display oracle diagnostics."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'results'
d=json.loads((OUT/'proposal_screen.json').read_text());base=np.load(ROOT/'experiments/local_presence/results/ps-data-small/maps-150.npz');lookup={str(n):i for i,n in enumerate(base['names'])}
selected=['330fb70f8303','6a87cd543012','ea6792c38efa','e7ece8e6125a','d854e128465a','9aec69d421d3']
try:
 font=ImageFont.truetype('DejaVuSans.ttf',17)
except OSError:
 font=ImageFont.load_default()
canvas=Image.new('RGB',(1200,6*290+50),'white');draw=ImageDraw.Draw(canvas)
for col,label in enumerate(['Image / ID','Ground truth (white=WM)','Base prediction','GT-selected oracle (<=2)']):draw.text((col*300+8,10),label,fill='black',font=font)
for row,name in enumerate(selected):
 item=next(x for x in d['per_image'] if x['image_name']==name);i=lookup[name];pred=base['probabilities'][i]>.5;gt=base['targets'][i]>.5;oracle=pred.copy();packed=np.load(OUT/'proposals'/f'{name}.npz')['packed'];masks=np.unpackbits(packed,axis=1).reshape(-1,128,128).astype(bool)
 actions=item['best_pair_actions'] if item['oracle_two']>item['oracle_one']+1e-12 else ([item['best_single_action']] if item['oracle_one']>item['base']+1e-12 else [])
 for action in actions:oracle[masks[action//2]]=action%2==0
 rgb=Image.open(ROOT/'data/train/images'/f'{name}.png').convert('RGB').resize((256,256));panels=[rgb]+[Image.fromarray(x.astype('uint8')*255).convert('RGB').resize((256,256),Image.Resampling.NEAREST) for x in [gt,pred,oracle]]
 y=50+row*290
 for col,p in enumerate(panels):canvas.paste(p,(col*300+8,y))
 labels=[name,f"WM area {item['true_area']:.2%}",f"mIoU {item['base']:.4f}",f"oracle {item['oracle_two']:.4f}"]
 for col,label in enumerate(labels):draw.text((col*300+8,y+260),label,fill='black',font=font)
canvas.save(OUT/'pixelseal_coverage_examples.png')
