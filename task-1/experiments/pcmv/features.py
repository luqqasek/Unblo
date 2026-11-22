"""Frozen, FP32 official-decoder counterfactuals and region-level carrier evidence."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import binary_erosion
from scipy.stats import rankdata
from .proposals import ring


def tensor(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float().cuda()/255


def signed_payload(bits):
    b=np.asarray(bits,dtype=np.float32)
    assert b.ndim==1 and np.isin(b,[0,1]).all(),'Raw binary payload required'
    return b*2-1


class Evidence:
    def __init__(self,wm,bias):
        self.wm=wm;self.bias=np.asarray(bias,np.float32);self.pooling_checked=False

    @torch.inference_mode()
    def decode(self,images):
        # Each original/counterfactual canvas or variable crop gets exactly the
        # official preprocessing once. No BF16 in watermark probes.
        out=[]
        for i in range(0,len(images),32):
            x=torch.cat([self.wm._resize(tensor(im)[None]) for im in images[i:i+32]])
            out.append(self.wm.det(x)[:,1:].float().cpu().numpy())
        return np.concatenate(out)

    @torch.inference_mode()
    def __call__(self,rgb,bits,proposals,base,local,obs):
        masks=np.stack([p[1] for p in proposals]);k=len(masks)
        images=[rgb];rings=[];capacity=[]
        for mask512,mask,_,_ in proposals:
            r=ring(mask);r512=np.repeat(np.repeat(r,4,0),4,1);rings.append(r)
            fill=np.rint(rgb[r512].mean(0) if r512.any() else rgb.mean((0,1))).astype(np.uint8)
            iso=np.where(mask512[...,None],rgb,fill);abl=np.where(mask512[...,None],fill,rgb)
            ri=np.where(r512[...,None],rgb,fill)
            yy,xx=np.where(mask512);crop=iso[yy.min():yy.max()+1,xx.min():xx.max()+1]
            # This checks bytes, not floating approximation of untouched RGB.
            assert np.array_equal(iso[mask512],rgb[mask512]) and np.array_equal(abl[~mask512],rgb[~mask512])
            images.extend([iso,abl,ri,crop]);capacity.append(float(obs[mask].sum()))
        logits=self.decode(images);raw=np.tanh(logits-self.bias)
        rng=np.random.default_rng(934);signs=rng.choice([-1.,1.],size=(65,len(bits))).astype(np.float32);signs[0]=signed_payload(bits)
        assert set(np.unique(signs[0])).issubset({-1.,1.})
        score=raw@signs.T/np.sqrt(np.maximum((raw*raw).sum(1,keepdims=True),1e-12))
        margin=raw@signs.T/len(bits)
        # The signed alignment is a feature. Empirical null is retained; no
        # independent-bit Gaussian significance is asserted.
        null_mean=score[:,1:].mean(1);null_sd=score[:,1:].std(1).clip(.05)
        calibrated=(score-null_mean[:,None])/null_sd[:,None]
        d=self.wm.features(tensor(rgb)[None])['per_position_logits'].float()
        if not self.pooling_checked:
            error=float(np.max(np.abs(d.mean((2,3))[0].cpu().numpy()-logits[0])))
            assert error<1e-4,('Frozen dense/global decoder mismatch',error)
            print('PCMV_DENSE_POOL_PARITY',error,flush=True);self.pooling_checked=True
        pooling=[]
        for m in masks:
            weight=F.interpolate(torch.tensor(m,device='cuda')[None,None].float(),size=d.shape[-2:],mode='area')
            wj=F.interpolate(torch.tensor(m*obs,device='cuda')[None,None].float(),size=d.shape[-2:],mode='area')
            pooling.append(torch.cat([(d*weight).sum((2,3))/weight.sum().clamp_min(1e-6),(d*wj).sum((2,3))/wj.sum().clamp_min(1e-6)],0).cpu().numpy())
        pooling=np.tanh(np.stack(pooling)-self.bias);pool_score=np.einsum('kpb,qb->kpq',pooling,signs)/np.sqrt(np.maximum((pooling*pooling).sum(2,keepdims=True),1e-12))
        cuts=np.quantile(obs,[.25,.5,.75]);quart=np.digitize(obs,cuts)
        values=[];wrong=[];keys=None;basic_indices=None
        for j,(m,r) in enumerate(zip(masks,rings)):
            f={};bkeys=[]
            def add(key,value,basic=False):
                f[key]=float(value)
                if basic:bkeys.append(key)
            yy,xx=np.where(m);a=float(m.mean());boundary=m&~binary_erosion(m)
            add('area',a,True);add('bbox_fill',m.sum()/((yy.max()-yy.min()+1)*(xx.max()-xx.min()+1)),True);add('perimeter_area',boundary.sum()/m.sum(),True)
            for name,array in [('base',base),('local',local)]:
                for where,region in [('inside',m),('ring',r)]:
                    v=array[region];add(name+'_'+where+'_mean',v.mean(),True);add(name+'_'+where+'_std',v.std(),True)
                add(name+'_contrast',array[m].mean()-array[r].mean(),True)
                add(name+'_boundary',array[boundary].mean(),True)
            add('obs_capacity',capacity[j]);add('obs_inside',obs[m].mean());add('obs_ring',obs[r].mean())
            for q in range(5):
                region=(quart==q) if q<4 else (quart>=2);inside=local[m&region];outside=local[r&region]
                add(f'q{q}_inside_n',len(inside));add(f'q{q}_outside_n',len(outside))
                diff=inside.mean()-outside.mean() if len(inside) and len(outside) else 0
                se=np.sqrt(inside.var()/len(inside)+outside.var()/len(outside)+1e-5) if len(inside) and len(outside) else 1
                add(f'q{q}_contrast',diff);add(f'q{q}_welch',np.clip(diff/se,-50,50))
            common=dict(f)
            def conditioned(q):
                v=dict(common);i=1+j*4
                for suffix,idx in [('global',0),('iso',i),('abl',i+1),('ring',i+2),('crop',i+3)]:
                    v['msg_'+suffix]=float(calibrated[idx,q]);v['margin_'+suffix]=float(margin[idx,q])
                v['msg_iso_ring']=float(calibrated[i,q]-calibrated[i+2,q]);v['drop']=float(margin[0,q]-margin[i+1,q]);v['drop_capacity']=v['drop']/max(capacity[j]/16384,1e-5)
                v['masked_pool']=float(pool_score[j,0,q]);v['jnd_masked_pool']=float(pool_score[j,1,q])
                v['global_bit_accuracy']=float(((logits[0]>0)==(signs[q]>0)).mean())
                return v
            f=conditioned(0);w=conditioned(1)
            if keys is None:keys=list(f);basic_indices=[keys.index(key) for key in bkeys]
            assert list(f)==keys;values.append(list(f.values()));wrong.append(list(w.values()))
        values=np.asarray(values,np.float32);wrong=np.asarray(wrong,np.float32)
        # Per-image rank features cannot expose GT or other images' information.
        for key in ['base_contrast','local_contrast','q4_contrast','msg_iso','msg_crop','drop_capacity']:
            column=keys.index(key);values=np.column_stack([values,rankdata(values[:,column])/k]);wrong=np.column_stack([wrong,rankdata(wrong[:,column])/k]);keys.append('rank_'+key)
            if key in ['base_contrast','local_contrast']:basic_indices.append(len(keys)-1)
        assert np.isfinite(values).all() and np.isfinite(wrong).all()
        return values.astype(np.float32),wrong.astype(np.float32),keys,basic_indices
