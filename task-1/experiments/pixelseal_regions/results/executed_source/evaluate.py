"""Known-geometry recognition, GT-free search, inner-only threshold calibration."""
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.native_localizer.common import atomic_json,rgb_tensor
from experiments.native_localizer.evaluate import image_metrics,gallery
from experiments.triforensic_wm.evaluate import summarize
from .geometry import candidates,tight,jitter,targets,draw


def score_boxes(model,tables,boxes,hw,chunk=16384):
    chunks=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for b in boxes.split(chunk):chunks.append(model.score(tables,b,hw).float())
    return torch.cat(chunks)


@torch.inference_mode()
def search(model,image,device,cache):
    x=rgb_tensor(image)[None].to(device)
    with torch.autocast('cuda',dtype=torch.bfloat16):f,o=model.encode(x)
    hw=f.shape[-2:];table=model.tables(f[0],o[0]);key=tuple(hw)
    if key not in cache:cache[key]=torch.tensor(candidates(*hw,step=2),device=device)
    boxes=cache[key];logits=score_boxes(model,table,boxes,hw);scores=logits.sigmoid().prod(1);top=scores.topk(min(16,len(scores))).indices
    # Refine coordinates independently at stride4; no raw-image interpolation.
    seeds=boxes[top].cpu().numpy();variants=[seeds]
    for coordinate in range(4):
        for delta in (-1,1):
            b=seeds.copy();b[:,coordinate]+=delta;variants.append(b)
    b=np.unique(np.concatenate(variants),axis=0);b[:,[0,2]]=b[:,[0,2]].clip(0,hw[1]);b[:,[1,3]]=b[:,[1,3]].clip(0,hw[0]);w=b[:,2]-b[:,0];h=b[:,3]-b[:,1];keep=(w>0)&(h>0)&(w*h<=820)&(w<=16*h)&(h<=16*w);b=torch.as_tensor(b[keep],device=device)
    s=score_boxes(model,table,b,hw).sigmoid().prod(1);idx=int(s.argmax());return b[idx].cpu().numpy(),float(s[idx]),table,boxes


@torch.inference_mode()
def recognition(model,bank,device,rank,world):
    model.eval();records=[]
    for i,r in enumerate(bank.rows['calibration']+bank.rows['audit']):
        if i%world!=rank:continue
        box=torch.tensor(tight(r['mask']),device=device)[None];pred=[]
        for name in ('clean','marked'):
            x=rgb_tensor(r[name])[None].to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):f,o=model.encode(x);z=model.score(model.tables(f[0],o[0]),box,r['mask'].shape)
            pred.append(float(z[0,0].float()))
        records.append({'id':r['id'],'source':r['source'],'kind':r['kind'],'role':r['role'],'clean':pred[0],'marked':pred[1],'short':min(r['original_crop_size']),'area':int(np.prod(r['original_crop_size'])),'zero_fraction':r['zero_fraction']})
    return records


def recognition_summary(rows):
    result={}
    groups={'all':rows,'real_clean':[r for r in rows if r['kind']=='real_clean'],'public':[r for r in rows if r['kind']=='public'],'width_le24':[r for r in rows if r['short']<=24]}
    for key,rr in groups.items():
        vals=[r[k] for r in rr for k in ('clean','marked')];result[key]={'n_pairs':len(rr),'auc':float(roc_auc_score([0,1]*len(rr),vals)) if rr else None,'paired_order_accuracy':float(np.mean([r['marked']>r['clean'] for r in rr])) if rr else None}
    return result


@torch.inference_mode()
def real_diagnostic(model,bank,device,rank,world):
    rows=[];rng=np.random.default_rng(2188)
    for i,name in enumerate(bank.split['validation']):
        if i%world!=rank:continue
        r=bank.real[name];y=r['mask128'];area=y.mean()
        if not 0<area<=.05:continue
        box=tight(y);x=rgb_tensor(r['image'])[None].to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):f,o=model.encode(x)
        table=model.tables(f[0],o[0]);b=torch.tensor(box,device=device)[None];positive=float(score_boxes(model,table,b,(128,128))[0,0]);w=box[2]-box[0];h=box[3]-box[1];neg=[]
        for _ in range(200):
            xx=int(rng.integers(129-w));yy=int(rng.integers(129-h))
            if not y[yy:yy+h,xx:xx+w].any():neg.append([xx,yy,xx+w,yy+h])
            if len(neg)==20:break
        if neg:
            zz=score_boxes(model,table,torch.tensor(neg,device=device),(128,128))[:,0].cpu().numpy();rows.append({'name':name,'area':float(area),'bbox_presence':positive,'background_scores':zz.tolist(),'within_image_rank':float((positive>zz).mean())})
    return rows


@torch.inference_mode()
def scan_real(model,bank,device,rank,world,basepath):
    z=np.load(basepath);baseline=dict(zip(z['names'].astype(str),z['probabilities']));cache={};records=[]
    names=bank.split['calibration']+bank.split['validation']
    for i,name in enumerate(names):
        if i%world!=rank:continue
        r=bank.real[name];p=baseline[name];base=image_metrics(p,r['mask128']);eligible=float((p>=.5).mean())<=.10
        if eligible:
            box,score,table,boxes=search(model,r['image'],device,cache);pred=p.copy();pred[draw(box)]=1.;after=image_metrics(pred,r['mask128']);overlap=targets(r['mask128'],boxes)[:,1];coverage=float(overlap.max())
        else:box=None;score=-1.;after=base;coverage=None
        records.append({'name':name,'role':'calibration' if name in bank.split['calibration'] else 'validation','box':box.tolist() if box is not None else None,'score':score,'eligible':eligible,'base':base,'after':after,'oracle_grid_iou':coverage})
        print('SCANNED',rank,name,round(score,4),flush=True)
    return records


def finish(rows,recognition_rows,diag,bank,out,step,basepath):
    calibration=[r for r in rows if r['role']=='calibration'];valid=[r for r in rows if r['role']=='validation'];thresholds=[0,.005,.01,.02,.03,.05,.08,.12,.2,.3,.5,.7,1.1];choices=[]
    for threshold in thresholds:
        chosen=[r['after'] if r['score']>=threshold else r['base'] for r in calibration]
        choices.append({'threshold':threshold,'miou':float(np.mean([r['miou'] for r in chosen])),'changes':sum(r['score']>=threshold for r in calibration)})
    selected=max(choices,key=lambda r:(r['miou'],r['threshold']));threshold=selected['threshold'];final=[];deltas=[]
    z=np.load(basepath);bp=dict(zip(z['names'].astype(str),z['probabilities']));maps=[];names=[];ys=[]
    for r in valid:
        applied=r['score']>=threshold;m=dict(r['after'] if applied else r['base'],image_name=r['name']);final.append(m);deltas.append(m['miou']-r['base']['miou']);p=bp[r['name']].copy()
        if applied:p[draw(r['box'])]=1.
        names.append(r['name']);maps.append(p);ys.append(bank.real[r['name']]['mask128'])
    deltas=np.array(deltas);rng=np.random.default_rng(42);ci=np.quantile(deltas[rng.integers(len(deltas),size=(5000,len(deltas)))].mean(1),[.025,.975]);summary=summarize(final);before=summarize([r['base'] for r in valid]);positive=int(sum(r['score']>=threshold for r in valid));gate=bool(deltas.mean()>=.005 and summary['fg_le1']['fg_iou']+summary['fg_1to5']['fg_iou']>before['fg_le1']['fg_iou']+before['fg_1to5']['fg_iou'] and summary['wm_over50']['clean_iou']>=before['wm_over50']['clean_iou']-.01)
    result={'step':step,'protocol':'One reused 57-image PixelSeal development fold; new classifier trained on90 real sources plus exact clean-region/public pairs;23 source-disjoint real inner calibration. Base EMA1561 saw all113 outer-training images; base-calibration optimism remains, specialist does not see23 calibration images. No full OOF.','baseline':before,'summary':summary,'delta':float(deltas.mean()),'ci95':ci.tolist(),'selected_threshold':threshold,'calibration':choices,'applied':positive,'candidate_for_test':gate,'recognition':recognition_summary(recognition_rows),'real_known_geometry':diag,'per_image':rows,'recognition_per_pair':recognition_rows}
    atomic_json(out/f'evaluation-{step}.json',result);np.savez_compressed(out/f'maps-{step}.npz',names=names,probabilities=maps,targets=ys)
    order=np.argsort(deltas);chosen=list(order[:2])+list(order[-3:])+[i for i,n in enumerate(names) if 0<ys[i].mean()<=.01][:2]
    for i in sorted(set(chosen)):
        name=names[i];gallery(out/f'example-{step}-{name}.png',bank.real[name]['image'],ys[i],maps[i],bp[name])
    print('FINAL',json.dumps({k:result[k] for k in ('step','delta','ci95','selected_threshold','applied','candidate_for_test','recognition')}),flush=True);return result
