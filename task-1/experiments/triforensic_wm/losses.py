import math
import torch
from torch.nn import functional as F

def sym_miou(z,y):
    p=z.float().sigmoid();y=y.float();scores=[];present=[]
    for a,b in ((p,y),(1-p,1-y)):
        inter=(a*b).flatten(1).sum(1);union=(a+b-a*b).flatten(1).sum(1)
        has=b.flatten(1).sum(1)>0;scores.append(has*(inter+1e-6)/(union+1e-6));present.append(has.float())
    return (1-(scores[0]+scores[1])/(present[0]+present[1]).clamp_min(1)).mean()

def focal(z,y):
    z=z.float();y=y.float();bce=F.binary_cross_entropy_with_logits(z,y,reduction='none');p=z.sigmoid();pt=p*y+(1-p)*(1-y)
    return ((1-pt).square()*bce).flatten(1).mean(1).mean()

def objective(out,y,pair_y):
    y=y.float();pair_y=pair_y.float();edge=(F.max_pool2d(y,3,1,1)+F.max_pool2d(-y,3,1,1)).clamp(0,1)
    obs=out['obs'][:,:1];median=obs.flatten(1).median(1).values[:,None,None,None];low=y*(obs<median)
    lp=out['l_pair'].float();patch=torch.logsumexp(lp.flatten(1),1)-math.log(lp.shape[-1]*lp.shape[-2])
    paired=F.binary_cross_entropy_with_logits(lp,pair_y)+F.binary_cross_entropy_with_logits(patch,pair_y.flatten(1).amax(1))
    terms={'seg_iou':sym_miou(out['logit_final'],y),'seg_focal':.30*focal(out['logit_final'],y),
           'base':.25*sym_miou(out['logit_base'],y),'local':.30*focal(out['l_loc'],y),
           'edge':.25*(focal(out['edge_logit'],edge)+sym_miou(out['edge_logit'],edge)),
           'veto_hard_negative':.15*(out['delta_clean']*low).mean(),'paired':.50*paired}
    return sum(terms.values()),terms
