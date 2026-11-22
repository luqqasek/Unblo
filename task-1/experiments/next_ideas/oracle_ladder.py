"""GT-informed decoder diagnostics; outputs must NEVER be inference proposals."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.native_localizer.common import atomic_json,rgb_tensor
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'next-ideas-runs/oracle';out.mkdir(parents=True,exist_ok=True)
    for family in ('pixelseal','trustmark','stegastamp'):
        if (out/f'{family}.json').exists():continue
        start=time.monotonic();wm=load_watermark(family,models_dir='checkpoints',device='cuda');ds=LocalisationDataset('data','train',methods=[family]);rows=[]
        messages=np.array([[int(c) for c in r['message']] for r in ds.meta]);signed=2*messages.astype(float)-1
        for i in range(len(ds)):
            it=ds[i];m=it['mask'].astype(bool);area=float(it['mask128'].mean());tiny=0<area<=.05;hole=.5<area<1
            if not (tiny or hole):continue
            x=rgb_tensor(it['image'])[None].cuda();mask=torch.tensor(m,device='cuda')[None,None];blur=F.avg_pool2d(F.pad(x,(8,8,8,8),mode='reflect'),17,1)
            yy,xx=np.where(m);box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)];x0,y0,x1,y1=box
            views={'full':x,'GT_bbox_direct_native':x[:,:,y0:y1,x0:x1],'outside_GT_blurred':torch.where(mask,x,blur),'GT_blurred':torch.where(mask,blur,x)}
            # A second outside-support replacement identifies boundary artifacts.
            views['outside_GT_mean']=torch.where(mask,x,x.mean((-2,-1),keepdim=True))
            row={'image_name':it['image_name'],'family':family,'true_area':area,'full_watermark':bool(it['full_watermark']),'bbox':box,'bbox_fill':float(m.sum()/((x1-x0)*(y1-y0))),'views':{}}
            for kind,value in views.items():
                logits=wm.logits(value)[0].detach().cpu().numpy();correct=int(np.sum((logits>0)==messages[i]));soft=signed@np.tanh(logits/2)/len(logits);null=np.delete(soft,i)
                row['views'][kind]={'correct_bits':correct,'bit_accuracy':correct/len(logits),'soft_agreement':float(soft[i]),'wrong_message_mean':float(null.mean()),'wrong_message_std':float(null.std()),'empirical_wrong_p':float((1+np.sum(null>=soft[i]))/(len(null)+1))}
            rows.append(row)
            if len(rows)%20==0:print('ORACLE_PROGRESS',family,len(rows),time.monotonic()-start,flush=True)
        summary={}
        for name,fn in [('fg_le1',lambda r:0<r['true_area']<=.01),('fg_1to5',lambda r:.01<r['true_area']<=.05),('holes',lambda r:r['true_area']>.5)]:
            rr=[r for r in rows if fn(r)];summary[name]={'n':len(rr)}
            for view in views:summary[name][view]={'mean_bit_accuracy':float(np.mean([r['views'][view]['bit_accuracy'] for r in rr])) if rr else None,'n_ge90':sum(r['views'][view]['bit_accuracy']>=.9 for r in rr)}
        atomic_json(out/f'{family}.json',{'family':family,'summary':summary,'per_image':rows,'seconds':time.monotonic()-start,'GT_used':True,'scope':'oracle diagnostic only; no proposals, threshold selection, or inference inputs','limits':'Failure in these views is not proof of absent signal. Blur/mean replacement changes decoder context and creates boundaries. Native resize is performed ONCE by unchanged official wrapper.'});print('ORACLE_DONE',family,json.dumps(summary),flush=True)
        del wm;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
