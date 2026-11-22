import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from experiments.native_localizer.common import atomic_json,rgb_tensor
from experiments.native_localizer.evaluate import image_metrics,gallery
from experiments.local_presence.train import summary
from experiments.local_presence.verification import verified
from experiments.small_zoom.proposals import boxmask
from .data import device_batch

def summarize(rows):
    result=summary(rows)
    filters={'all':lambda r:True,'fg_le1':lambda r:0<r['true_area']<=.01,'fg_1to5':lambda r:.01<r['true_area']<=.05,'fg_5to20':lambda r:.05<r['true_area']<=.2,'fg_gt20':lambda r:r['true_area']>.2,
             'clean_le1':lambda r:0<1-r['true_area']<=.01,'clean_1to5':lambda r:.01<1-r['true_area']<=.05,'wm_over50':lambda r:.5<r['true_area']<1,
             'fully_clean':lambda r:r['true_area']==0,'fully_marked':lambda r:r['true_area']==1}
    for kind in ('rectangular','irregular'):
        filters['fg_1to5_'+kind]=lambda r,k=kind:.01<r['true_area']<=.05 and r.get('shape')==k
    for key,fn in filters.items():
        rr=[r for r in rows if fn(r)];g=summary(rr)['all'];g['fg_miss']=sum(r['fg_iou']==0 and r['true_area']>0 for r in rr);g['clean_miss']=sum(r['clean_iou']==0 and r['true_area']<1 for r in rr)
        g['defined_counts']={k:sum(r.get(k) is not None for r in rr) for k in ('fg_ap','clean_ap','roc_auc')};result[key]=g
    return result

def old_rectangles(root):
    path=Path(root)/'small-zoom-finalscale/trustmark/proposals.json'
    result={}
    if path.exists():
        for row in json.loads(path.read_text()):
            if row['correct_top'] and verified(row['correct_top'][0]):result[row['image_name']]=row['correct_top'][0]['box']
    return result

@torch.no_grad()
def real_evaluate(model,engine,out,step,baseline=False,rectangles=None):
    model.eval();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    probs=[];raws=[];locals_=[];cleans=[];ys=[];names=[];rows=[];rawrows=[];localrows=[]
    for start in range(0,len(engine.valid),8):
        ids=engine.valid[start:start+8];batch=device_batch(engine.stack([engine.real_full(i) for i in ids]))
        if baseline:
            z=model(batch['x'],batch['bits'])['logits'][:,None];pred={'logit_final':z,'l_loc':z,'delta_clean':torch.zeros_like(z)}
        else:pred=model(batch['x'],batch['bits'],batch['npmap'])
        pp=pred['logit_final'][:,0].float().sigmoid().cpu().numpy();lp=pred['l_loc'][:,0].float().sigmoid().cpu().numpy();dc=pred['delta_clean'][:,0].float().cpu().numpy()
        for j,i in enumerate(ids):
            item=engine.real[i];y=item['mask128'];name=item['image_name'];p=pp[j].copy();raw=p.copy()
            if engine.family=='trustmark' and rectangles and name in rectangles:
                box=rectangles[name];area=(box[2]-box[0])*(box[3]-box[1])/512**2
                if baseline:p[boxmask(box)]=1
                elif area<=.15:p[boxmask(box)&(dc[j]<=2.)]=1
            yb=y.astype(bool);yy,xx=np.where(yb);fill=float(yb.sum()/((yy.max()-yy.min()+1)*(xx.max()-xx.min()+1))) if len(yy) else 0.
            for value,dest in ((p,rows),(raw,rawrows),(lp[j],localrows)):
                r=image_metrics(value,y);r.update(image_name=name,shape='rectangular' if fill>=.98 else 'irregular',bbox_fill=fill);dest.append(r)
            probs.append(p);raws.append(raw);locals_.append(lp[j]);cleans.append(dc[j]);ys.append(y);names.append(name)
    record={'step':step,'summary':summarize(rows),'raw_summary':summarize(rawrows),'local_summary':summarize(localrows),'per_image':rows,'raw_per_image':rawrows,'local_per_image':localrows,
            'protocol':'fixed fold1; threshold0.5; no GT at inference; existing TM boxes <=15% and delta_clean<=2 for new model; baseline existing rects unchanged',
            'ap':'sklearn non-interpolated average precision per image; class-absent excluded','validation_used_for':'screening gate and engineering, not an unbiased estimate after repeated experiments'}
    atomic_json(out/f'real-{step}.json',record)
    np.savez_compressed(out/f'maps-{step}.npz',names=names,probabilities=np.array(probs),raw_probabilities=np.array(raws),local_probabilities=np.array(locals_),delta_clean=np.array(cleans),targets=np.array(ys))
    print('REAL_VALIDATION',step,json.dumps({k:record[k] for k in ('summary','local_summary')}),flush=True)
    return record

@torch.no_grad()
def paired_evaluate(model,engine,out,step):
    """Interior evidence uses full RF29 support, excludes padding and paste seams."""
    model.eval();rng=np.random.default_rng(8821);records=[]
    indices=engine.dev.buckets[1]+engine.dev.buckets[2][:128]
    for start in range(0,len(indices),32):
        crops=[engine.dev.crop(i,rng,diagnostic=True) for i in indices[start:start+32]]
        x=torch.stack([rgb_tensor(im) for a,b,*_ in crops for im in (a,b)]).cuda()
        _,z,obs=model.local_forward(x);prob=z[:,0].float().sigmoid().cpu().numpy();obs=obs[:,0].float().cpu().numpy()
        for j,(_,_,y,pure,meta) in enumerate(crops):
            record={'index':meta['index'],'source_id':meta['source_id'],'bucket':meta['bucket'],'operation':meta['operation'],'interior_cells':int(pure.sum()),'jnd':float(obs[j*2+1].mean()),
                    'clean_score':float(prob[j*2].mean()),'marked_score':float(prob[j*2+1].mean()),
                    'interior_clean':float(prob[j*2][pure].mean()) if pure.any() else None,
                    'interior_marked':float(prob[j*2+1][pure].mean()) if pure.any() else None}
            records.append(record)
    def auc(rr,interior=False):
        keys=('interior_clean','interior_marked') if interior else ('clean_score','marked_score');r=[s for s in rr if s[keys[0]] is not None]
        values=[s[k] for s in r for k in keys];return {'n_pairs':len(r),'auc':float(roc_auc_score([0,1]*len(r),values)) if r else None}
    tiny=[r for r in records if r['bucket']==1];bounds=np.quantile([r['jnd'] for r in tiny],[0,.25,.5,.75,1])
    result={'step':step,'tiny_all':auc(tiny),'tiny_interior':auc(tiny,True),'remnants':auc([r for r in records if r['bucket']==2]),
            'jnd_quartiles':[{**auc([r for r in tiny if bounds[q]<=r['jnd']<=bounds[q+1]],True),'jnd_range':bounds[q:q+2].tolist()} for q in range(4)],'per_pair':records,
            'interior_definition':'RF29 fully within watermark support and64px crop; widths<29 have no measurable interior, excluded and counted'}
    atomic_json(Path(out)/f'paired-{step}.json',result);print('PAIRED_VALIDATION',step,json.dumps({k:v for k,v in result.items() if k!='per_pair'}),flush=True)
    return result

@torch.no_grad()
def synthetic_evaluate(model,engine,out,step):
    model.eval();indices=[i for b in range(5) for i in engine.dev.buckets[b][:16]];rows=[];intact=[]
    for start in range(0,len(indices),8):
        ids=indices[start:start+8];batch=device_batch(engine.stack([engine.dev.full(i) for i in ids]));o=model(batch['x'],batch['bits'],batch['npmap'])
        p=o['logit_final'][:,0].sigmoid().float().cpu().numpy();v=o['delta_clean'][:,0].float().cpu().numpy()
        for j,i in enumerate(ids):
            r=image_metrics(p[j],engine.dev.get(i,'target'));r.update(index=engine.dev.rows[i]['index'],operation=engine.dev.rows[i]['operation']);rows.append(r)
            if r['operation']=='intact_low_observability':intact.append({'index':r['index'],'false_clean_area':float((p[j]<.5).mean()),'veto_above2_area':float((v[j]>2).mean())})
    result={'step':step,'summary':summarize(rows),'per_image':rows,'intact_false_veto':intact};atomic_json(Path(out)/f'synthetic-{step}.json',result);return result
