"""Bounded no-training geometry screen: PS fold1, partial GT-free proposal set, exact oracle pairs."""
import os,sys,site,time,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];os.chdir(ROOT);OUT=HERE/'results';OUT.mkdir(parents=True,exist_ok=True)
vendor=ROOT/'experiments/pretrained_forensics/results';sys.path[:0]=[str(vendor/'deps'),str(vendor/'vendor/sam2'),str(ROOT)];site.addsitedir(str(vendor/'deps'))
import numpy as np
import torch,cv2
from PIL import Image
from scipy import ndimage
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from wmloc.utils import downsample_mask,miou_image

def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
 with p.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
def proposals(rgb,base,generator):
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):sam=generator.generate(rgb)
 candidates=[]
 # Fixed quality/area quotas, never GT-ranked. Single grid + SAM multimask outputs.
 for lo,hi in [(0,.01),(.01,.05),(.05,.2),(.2,1.01)]:
  group=sorted([p for p in sam if lo<p['area']/512**2<=hi],key=lambda p:p['predicted_iou']*p['stability_score'],reverse=True)[:10]
  for p in group:candidates.append((downsample_mask(p['segmentation']).astype(bool),'sam2'))
 rectangles=[]
 for p in sorted(sam,key=lambda p:p['predicted_iou']*p['stability_score'],reverse=True):
  x,y,w,h=map(int,p['bbox']);rectangles.append((x,y,x+w,y+h))
 edges=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),60,120);contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
 for c in sorted(contours,key=cv2.arcLength if False else lambda c:cv2.arcLength(c,True),reverse=True)[:80]:
  x,y,w,h=cv2.boundingRect(c)
  if min(w,h)>=4 and max(w/h,h/w)<=16 and w*h<.5*512**2:rectangles.append((x,y,x+w,y+h))
 for x0,y0,x1,y1 in rectangles[:20]:
  mask=np.zeros((512,512),bool);mask[max(y0,0):min(y1,512),max(x0,0):min(x1,512)]=True;candidates.append((downsample_mask(mask).astype(bool),'rectangle'))
 for value in (False,True):
  labels,n=ndimage.label(base==value,np.ones((3,3)));sizes=np.bincount(labels.ravel());order=np.argsort(sizes[1:])[::-1][:10]+1
  for k in order:candidates.append((labels==k,'base_component'))
 selected=[];origins=[];hashes=set()
 for mask,origin in candidates:
  if not mask.any():continue
  packed=np.packbits(mask).tobytes()
  if packed in hashes:continue
  if any((mask&old).sum()/max((mask|old).sum(),1)>.95 for old in selected):continue
  selected.append(mask);origins.append(origin);hashes.add(packed)
  if len(selected)==80:break
 return np.stack(selected),origins,len(sam)

def oracle(base,target,masks):
 t=target.astype(bool);b=base.astype(bool);counts=np.array([(b&t).sum(),(b&~t).sum(),(~b&t).sum(),(~b&~t).sum()],float)
 deltas=[]
 for s in masks:
  a=(s&~b&t).sum();d=(s&~b&~t).sum();deltas.append([a,d,-a,-d])
  a=(s&b&t).sum();d=(s&b&~t).sum();deltas.append([-a,-d,a,d])
 d=np.array(deltas)
 def value(c):
  tp,fp,fn,tn=np.moveaxis(c,-1,0);uf=tp+fp+fn;uc=tn+fp+fn
  return .5*(np.divide(tp,uf,out=np.ones_like(tp),where=uf>0)+np.divide(tn,uc,out=np.ones_like(tn),where=uc>0))
 base_score=float(value(counts));singles=value(counts+d);best_single=max(base_score,float(singles.max()))
 flat=masks.reshape(len(masks),-1).astype(np.float32);disjoint=flat@flat.T==0;compatible=np.repeat(np.repeat(disjoint,2,axis=0),2,axis=1)
 joint=value(counts+d[:,None]+d[None,:]);joint[~compatible]=-np.inf
 best_two=max(best_single,float(joint.max()))
 assert best_two+1e-12>=best_single>=base_score-1e-12
 return {'base':base_score,'oracle_one':best_single,'oracle_two':best_two,'best_single_action':int(np.argmax(singles)),'best_pair_actions':[int(x) for x in np.unravel_index(np.argmax(joint),joint.shape)] if np.isfinite(joint).any() else None}

def main():
 torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert '3070' in torch.cuda.get_device_name(0)
 path=ROOT/'experiments/local_presence/results/ps-data-small/maps-150.npz';p=np.load(path);assert len(p['names'])==57
 expected=np.mean([miou_image(a>=.5,y) for a,y in zip(p['probabilities'],p['targets'])]);assert abs(expected-.570883)<1e-6,expected
 checkpoint=vendor/'vendor/DetectiveSAM/sam2configs/sam2.1_hiera_base_plus.pt';model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(checkpoint),device='cuda',apply_postprocessing=False);model.requires_grad_(False)
 generator=SAM2AutomaticMaskGenerator(model,points_per_side=16,points_per_batch=16,pred_iou_thresh=.6,stability_score_thresh=.9,crop_n_layers=0,min_mask_region_area=0,output_mode='binary_mask')
 config={'scope':'PS fixed57-image geometry-only oracle; incomplete PCMV Stage0','base':'#3005 PS unchanged from #2675','baseline_sha256':sha(path),'sam_checkpoint_sha256':sha(checkpoint),'points_per_side':16,'points_per_batch':16,'pred_iou_thresh':.6,'stability_score_thresh':.9,'crop_n_layers':0,'max_candidates':80,'quota':'40 SAM area-stratified,20 rectangles,20 base components before dedup','missing':['forensic-prompted SAM','JND local extrema/watershed','NRGMark boxes','counterfactual features','learned acceptance'],'composition':'Direct toggles on official128 grid; proposals majority-downsampled from512. Exact enumeration of at most2 disjoint actions, not sum of individual gains.','GT_used_for':'metrics and oracle action selection only, after proposal generation/capping','gpu':torch.cuda.get_device_name(0)};save(OUT/'proposal_screen_config.json',config)
 cache=OUT/'proposals';cache.mkdir(exist_ok=True);rows=[];start=time.monotonic()
 for name,prob,target in zip(p['names'].astype(str),p['probabilities'],p['targets']):
  dest=cache/f'{name}.npz';base=prob>=.5;began=time.monotonic()
  if dest.exists():
   saved=np.load(dest);masks=np.unpackbits(saved['packed'],axis=1)[:,:128**2].reshape(-1,128,128).astype(bool);origins=saved['origins'].astype(str).tolist();raw=int(saved['raw_sam'])
  else:
   rgb=np.array(Image.open(ROOT/'data/train/images'/f'{name}.png').convert('RGB'));masks,origins,raw=proposals(rgb,base,generator)
   np.savez_compressed(dest,packed=np.packbits(masks.reshape(len(masks),-1),axis=1),origins=origins,raw_sam=raw)
  values=oracle(base,target,masks);minority=target.astype(bool) if target.mean()<.5 else ~target.astype(bool)
  overlaps=np.sum(masks&minority,axis=(1,2))/np.maximum(np.sum(masks|minority,axis=(1,2)),1)
  row={'image_name':name,'true_area':float(target.mean()),'n_proposals':len(masks),'n_raw_sam':raw,'minority_best_proposal_iou':float(overlaps.max()),'seconds':time.monotonic()-began,**values};rows.append(row)
  save(OUT/'proposal_screen_progress.json',{'done':len(rows),'total':57,'seconds':time.monotonic()-start});print('PROPOSALS',len(rows),57,'seconds',round(time.monotonic()-start,1),'gain',round(values['oracle_two']-values['base'],4),flush=True)
 summary={}
 for group,fn in [('all',lambda r:True),('fg_le1',lambda r:0<r['true_area']<=.01),('fg_1to5',lambda r:.01<r['true_area']<=.05),('clean_le1',lambda r:0<1-r['true_area']<=.01),('clean_1to5',lambda r:.01<1-r['true_area']<=.05)]:
  rr=[r for r in rows if fn(r)];summary[group]={'n':len(rr),**{k:float(np.mean([r[k] for r in rr])) if rr else None for k in ('base','oracle_one','oracle_two','minority_best_proposal_iou')},'proposal_recall_iou50':sum(r['minority_best_proposal_iou']>=.5 for r in rr)}
 save(OUT/'proposal_screen.json',{'config':config,'summary':summary,'per_image':rows,'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'trained_model':False,'PCMV_gain_measured':False});print('SCREEN_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
