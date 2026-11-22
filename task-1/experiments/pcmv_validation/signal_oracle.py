"""Frozen PixelSeal signal diagnostic with GT supports and matched translated controls. No learned scorer."""
import os,sys,json,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT));OUT=HERE/'results';OUT.mkdir(exist_ok=True)
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from scipy import ndimage
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert '3070' in torch.cuda.get_device_name(0)
 wm=load_watermark('pixelseal',ROOT/'checkpoints','cuda');start=time.monotonic()
 def maps(x,size):
  if size!=512:x=F.interpolate(x,(size,size),mode='bilinear',align_corners=False,antialias=True)
  f=wm.det.pixel_decoder.output_upscaling(wm.det.convnext(wm.det.preprocess(x)))
  h=wm.det.pixel_decoder.linear
  return F.conv2d(f,h.weight[:,:,None,None],h.bias)[:,1:]
 def tensor(a):return torch.from_numpy(np.array(a)).permute(0,3,1,2).cuda().float()/255
 data=ROOT/'experiments/rtx_all_information/results/data/pixelseal';records=json.loads((data/'records.json').read_text());clean=np.load(data/'clean.npy',mmap_mode='r')
 ids=np.array([i for i,r in enumerate(records) if r['partition']=='train']);ids=np.random.default_rng(42).permutation(ids)[:200];bias={}
 for size in (256,512):
  values=[]
  for j in range(0,len(ids),8):values.append(maps(torch.from_numpy(np.array(clean[ids[j:j+8]])).cuda().float()/255,size).mean((2,3)).cpu().numpy())
  bias[size]=np.concatenate(values).mean(0)
 np.savez_compressed(OUT/'signal_oracle_bias.npz',native256=bias[256],direct512=bias[512],source_ids=[records[int(i)]['source_id'] for i in ids])
 first=torch.from_numpy(np.array(clean[ids[:1]])).cuda().float()/255
 identity=float((maps(first,256).mean((2,3))-wm.logits(first)).abs().max());assert identity<1e-4,identity
 metadata=json.loads((ROOT/'data/train/metadata.json').read_text());allrows=[]
 for index,r in enumerate([r for r in metadata if r['method']=='pixelseal']):
  gt=np.array(Image.open(ROOT/'data/train/masks'/f"{r['image_name']}.png"))>127;area=downsample_mask(gt).mean()
  if not (0<area<=.05 or 0<1-area<=.05):continue
  s=gt if area<=.05 else ~gt;kind='fg_le5' if area<=.05 else 'clean_le5';polarity=1 if area<=.05 else -1
  rgb=np.array(Image.open(ROOT/'data/train/images'/f"{r['image_name']}.png").convert('RGB'));x=tensor(rgb[None]);jnd=wm.model.attenuation.heatmaps(x).mean(1)[0].cpu().numpy()
  yy,xx=np.where(s);y0,x0=yy.min(),xx.min();h,w=yy.max()-y0+1,xx.max()-x0+1;cy,cx=yy-y0,xx-x0;target_jnd=float(jnd[s].mean());candidates=[];seen=set();rng=np.random.default_rng(9100+index)
  for _ in range(300):
   dy=int(rng.integers(0,513-h));dx=int(rng.integers(0,513-w))
   if (dy,dx) in seen:continue
   seen.add((dy,dx));ny,nx=cy+dy,cx+dx
   if s[ny,nx].mean()>.01:continue
   mean=float(jnd[ny,nx].mean());gap=abs(mean-target_jnd)/max(target_jnd,1e-6)
   if gap>.5:continue
   candidates.append((gap,dy,dx))
  candidates.sort();candidates=candidates[:5]
  if not candidates:
   allrows.append({'image_name':r['image_name'],'kind':kind,'excluded':'no pure translated area-preserving control with JND mean within50%','true_area':float(area)});continue
  masks=[s]
  for gap,dy,dx in candidates:
   q=np.zeros_like(s);q[cy+dy,cx+dx]=True;masks.append(q)
  views=[];weights=[]
  for q in masks:
   distance=ndimage.distance_transform_edt(~q);v=distance[~q];threshold=np.partition(v,min(int(q.sum())-1,len(v)-1))[min(int(q.sum())-1,len(v)-1)];ring=(distance>0)&(distance<=threshold)
   mean=rgb[ring].mean(0).astype(np.float32)/255;source=rgb.astype(np.float32)/255
   # Keep float pixels unchanged inside supports. Fills intentionally need not be uint8.
   views.extend([np.where(q[:,:,None],source,mean),np.where(ring[:,:,None],source,mean),np.where(q[:,:,None],mean,source)])
   weights.append(float(jnd[q].sum()))
  v=torch.from_numpy(np.stack(views)).permute(0,3,1,2).cuda();sgn=np.array([2*int(c)-1 for c in r['message']],np.float32);wrong=rng.choice([-1.,1.],size=(256,256)).astype(np.float32);messages=np.vstack([sgn,wrong]);scores={}
  for size in (256,512):
   full=maps(x,size);glob=full.mean((2,3))[0].cpu().numpy();outputs=[]
   for j in range(0,len(v),8):outputs.append(maps(v[j:j+8],size).cpu().numpy())
   output=np.concatenate(outputs);fullnp=full[0].cpu().numpy();b=bias[size]
   def z(logits):
    a=np.tanh(logits-b);return a@messages.T/max(float(np.linalg.norm(a)),1e-12)
   methods={k:[] for k in ('iso_z','pool_z','iso_pool_z','iso_ring_contrast','ablation_raw','ablation_jnd')}
   for k,q in enumerate(masks):
    qsmall=F.interpolate(torch.from_numpy(q.astype(np.float32))[None,None],full.shape[-2:],mode='area')[0,0].numpy();qsmall/=max(float(qsmall.sum()),1e-12)
    iso,ring,abl=output[k*3:k*3+3];zi=z(iso.mean((1,2)));zr=z(ring.mean((1,2)));drop=(np.tanh(glob-b)-np.tanh(abl.mean((1,2))-b))@messages.T/256
    vals={'iso_z':zi,'pool_z':z((fullnp*qsmall).sum((1,2))),'iso_pool_z':z((iso*qsmall).sum((1,2))),'iso_ring_contrast':zi-zr,'ablation_raw':drop,'ablation_jnd':drop/max(weights[k],1e-6)}
    for key,value in vals.items():methods[key].append(value*polarity)
   for key,value in methods.items():
    arr=np.stack(value);diff=arr[0:1]-arr[1:];rank=((diff>0)+.5*(diff==0)).mean(0)
    scores[f'{size}_{key}']={'paired_ranking_true':float(rank[0]),'paired_ranking_wrong':rank[1:].tolist(),'true_positive_score':float(arr[0,0]),'true_negative_scores':arr[1:,0].tolist()}
  allrows.append({'image_name':r['image_name'],'kind':kind,'true_area':float(area),'controls':len(candidates),'jnd_relative_gaps':[float(c[0]) for c in candidates],'jnd_capacity':weights,'scores':scores})
  save(OUT/'signal_oracle_progress.json',{'rows':len(allrows),'evaluated':sum('scores' in q for q in allrows),'seconds':time.monotonic()-start});print('SIGNAL',len(allrows),r['image_name'],kind,'seconds',round(time.monotonic()-start,1),flush=True)
 summary={}
 for kind in ('fg_le5','clean_le5'):
  rr=[r for r in allrows if r['kind']==kind];valid=[r for r in rr if 'scores' in r];part={'n_total':len(rr),'n_evaluated':len(valid),'n_excluded':len(rr)-len(valid),'methods':{}}
  for key in valid[0]['scores'] if valid else []:
   true=np.array([r['scores'][key]['paired_ranking_true'] for r in valid]);wrong=np.array([r['scores'][key]['paired_ranking_wrong'] for r in valid]).mean(0)
   boot=np.random.default_rng(42).choice(true,(5000,len(true)),replace=True).mean(1)
   part['methods'][key]={'same_image_matched_ranking_auc':float(true.mean()),'image_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'wrong_message_mean':float(wrong.mean()),'wrong_message_95range':np.quantile(wrong,[.025,.975]).tolist(),'empirical_wrong_p_unadjusted':float((1+(wrong>=true.mean()).sum())/257)}
  if valid:
   nullmax=np.max(np.array([[r['scores'][k]['paired_ranking_wrong'] for r in valid] for k in valid[0]['scores']]).mean(1),axis=0);best=max(v['same_image_matched_ranking_auc'] for v in part['methods'].values());part['max_over12_methods_wrong_p']=float((1+(nullmax>=best).sum())/257)
  summary[kind]=part
 save(OUT/'signal_oracle.json',{'scope':'GT-support oracle signal diagnostic on PixelSeal; no learned verifier, no deployable proposal AUC, no local forensic branch','calibration':'200 exact clean synthetic TRAIN covers only; per-bit raw-logit means fitted separately for native256 and direct512','clean_source_ids':[records[int(i)]['source_id'] for i in ids],'full_pool_identity_max_error':identity,'views':'256 follows original wrapper;512 deliberately bypasses wrapper resize and is a distribution-shift diagnostic','negative_controls':'up to5 same-shape/area translated supports per image; <=1% true-minority overlap; JND mean within50%; selected without decoder scores','labels_used':True,'mask_grid':512,'wrong_messages':256,'methods_tested':12,'summary':summary,'per_image':allrows,'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
 print('SIGNAL_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
