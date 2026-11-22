"""GT-free SAM2, local-contrast, base-component and constrained-rectangle proposals."""
import numpy as np
from scipy.ndimage import label,gaussian_filter,maximum_filter,distance_transform_edt,binary_dilation,binary_erosion
import torch
from experiments.local_presence.data import down

class Proposals:
    def __init__(self,checkpoint):
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(checkpoint),device='cuda',mode='eval',apply_postprocessing=False)
        self.auto=SAM2AutomaticMaskGenerator(model,points_per_side=12,points_per_batch=64,crop_n_layers=0,min_mask_region_area=0,output_mode='binary_mask',pred_iou_thresh=.75,stability_score_thresh=.90,use_m2m=False,multimask_output=True)
        self.prompt=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
    @torch.inference_mode()
    def __call__(self,rgb,base,local,obs,limit=72):
        assert rgb.dtype==np.uint8 and rgb.shape==(512,512,3)
        candidates=[]
        def add(m,source,quality=0.):
            m=np.asarray(m,bool)
            if m.shape==(128,128):m=np.repeat(np.repeat(m,4,0),4,1)
            assert m.shape==(512,512)
            y=down(m).astype(bool);area=float(y.mean())
            if .0008<=area<=.65:candidates.append((m,y,source,float(quality)))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            objects=self.auto.generate(rgb.copy())
        # Keep several object scales, rather than letting largest masks occupy the bank.
        for lo,hi in [(0,.01),(.01,.05),(.05,.2),(.2,.65)]:
            group=[r for r in objects if lo<r['area']/512**2<=hi]
            for r in sorted(group,key=lambda r:r['predicted_iou']*r['stability_score'],reverse=True)[:8]:add(r['segmentation'],'sam_auto',r['predicted_iou'])
        smooth=gaussian_filter(local.astype(float),1);surround=gaussian_filter(smooth,5);contrast=(smooth-surround)/np.sqrt(np.maximum(gaussian_filter((smooth-surround)**2,5),1e-4))
        seeds=[]
        for sign in (1,-1):
            value=sign*contrast;peaks=(maximum_filter(value,13)==value);yy,xx=np.where(peaks)
            rank=np.argsort(value[yy,xx])[::-1]
            for j in rank[:4]:seeds.append((int(xx[j]),int(yy[j]),sign))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            self.prompt.set_image(rgb.copy())
            for x,y,sign in seeds:
                masks,quality,_=self.prompt.predict(point_coords=np.array([[x*4+2,y*4+2]],np.float32),point_labels=np.array([1],np.int32),normalize_coords=True,multimask_output=True)
                for j in np.argsort(quality)[-2:]:add(masks[j],'sam_forensic_prompt',quality[j])
        for threshold in (.3,.5,.7):
            for m in (base>threshold,base<threshold):
                cc,n=label(m);sizes=np.bincount(cc.ravel());ids=np.argsort(sizes[1:])[::-1][:3]+1
                for idx in ids:
                    region=cc==idx;add(region,'base_component',.6)
        for x,y,sign in seeds:
            for threshold in (.5,1.):
                cc,n=label(sign*contrast>threshold);idx=cc[y,x]
                if idx:add(cc==idx,'contrast_component',.5)
            # Axis-aligned boxes, dimensions chosen without any GT support.
            for w,h in ((8,8),(16,8),(8,32),(32,8)):
                x0=int(np.clip(x-w//2,0,128-w));y0=int(np.clip(y-h//2,0,128-h));m=np.zeros((128,128),bool);m[y0:y0+h,x0:x0+w]=1;add(m,'local_rectangle',.3)
        # Bboxes of semantic/components supply edge-aligned anisotropic candidates.
        for m,y,source,q in list(candidates)[:48]:
            yy,xx=np.where(y);w=int(xx.max()-xx.min()+1);h=int(yy.max()-yy.min()+1)
            if max(w/h,h/w)<=16:
                box=np.zeros((128,128),bool);box[yy.min():yy.max()+1,xx.min():xx.max()+1]=1;add(box,'proposal_bbox',q-.1)
        # Round-robin sources prevents a semantic-only proposal bank.
        groups={s:sorted([r for r in candidates if r[2]==s],key=lambda r:r[3],reverse=True) for s in ('sam_auto','sam_forensic_prompt','base_component','contrast_component','local_rectangle','proposal_bbox')};selected=[]
        for index in range(max(map(len,groups.values()),default=0)):
            for source,rr in groups.items():
                if index>=len(rr):continue
                item=rr[index];y=item[1]
                if any(np.logical_and(y,a[1]).sum()/np.logical_or(y,a[1]).sum()>.90 for a in selected):continue
                selected.append(item)
                if len(selected)>=limit:break
            if len(selected)>=limit:break
        if not selected:
            m=np.zeros((512,512),bool);m[224:288,224:288]=1;selected=[(m,down(m).astype(bool),'fixed_fallback',0.)]
        return selected

def ring(mask):
    m=np.asarray(mask,bool);distance=distance_transform_edt(~m);outside=distance[~m];k=min(int(m.sum()),len(outside))
    if k==0:return np.zeros_like(m)
    radius=np.partition(outside,k-1)[k-1]
    return (~m)&(distance<=radius)

def metrics(mask,gt):
    values=[]
    for c in (False,True):
        a=mask==c;b=gt==c;union=np.logical_or(a,b).sum();values.append(float(np.logical_and(a,b).sum()/union) if union else 1.)
    return sum(values)/2

def action_targets(base,masks,gt):
    before=metrics(base,gt);result=[]
    for m in masks:
        a=base.copy();a[m]=True;b=base.copy();b[m]=False
        result.append([metrics(a,gt)-before,metrics(b,gt)-before])
    return np.asarray(result,np.float32)

def compose(base,masks,gains,threshold,max_toggles=2):
    result=base.copy();occupied=np.zeros_like(base);applied=[]
    for flat in np.argsort(np.asarray(gains).ravel())[::-1]:
        i,action=divmod(int(flat),2);gain=float(gains[i,action]);m=masks[i]
        if gain<=threshold:break
        if np.any(occupied&m) or np.all(result[m]==(action==0)):continue
        result[m]=action==0;occupied|=m;applied.append({'proposal':i,'action':'add' if action==0 else 'carve','predicted_gain':gain})
        if len(applied)>=max_toggles:break
    return result,applied


def exact_oracle(base,masks,gt):
    """Exact <=2 disjoint toggles via additive confusion counts, NOT additive IoU."""
    base=np.asarray(base,bool);gt=np.asarray(gt,bool);masks=np.asarray(masks,bool)
    counts=np.array([((~base)&(~gt)).sum(),(base&(~gt)).sum(),((~base)&gt).sum(),(base&gt).sum()],np.float64)
    changes=[]
    for m in masks:
        tn=(m&~base&~gt).sum();fp=(m&base&~gt).sum();fn=(m&~base&gt).sum();tp=(m&base&gt).sum()
        changes.extend([[-tn,tn,-fn,fn],[fp,-fp,tp,-tp]])
    changes=np.asarray(changes)
    def score(c):
        tn,fp,fn,tp=np.moveaxis(c,-1,0);uc=tn+fp+fn;uw=tp+fp+fn
        clean=np.divide(tn,uc,out=np.ones_like(tn),where=uc!=0);wm=np.divide(tp,uw,out=np.ones_like(tp),where=uw!=0)
        return (clean+wm)/2
    one=score(counts+changes);two=score(counts+changes[:,None,:]+changes[None,:,:])
    packed=np.packbits(masks.reshape(len(masks),-1),axis=1);disjoint=~np.bitwise_and(packed[:,None,:],packed[None,:,:]).any(-1);eligible=np.repeat(np.repeat(disjoint,2,0),2,1)
    baseline=metrics(base,gt);best_one=max(baseline,float(one.max()));best_two=max(best_one,float(np.max(two[eligible])) if eligible.any() else baseline)
    return {'base':baseline,'one_toggle':best_one,'two_disjoint_toggles':best_two}
