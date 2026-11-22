"""Read-only input fidelity and cache audit before unified optimizer updates."""
import numpy as np
import torch
from experiments.native_localizer.common import rgb_tensor,atomic_json
from .model import observability

@torch.no_grad()
def fidelity(model,engine,path):
    model.eval();rng=np.random.default_rng(732);images=[];meta=[]
    for bucket in (1,2):
        for i in engine.dev.buckets[bucket][:8]:
            a,b,y=engine.dev.crop(i,rng);images.extend((a,b));d=b.astype(np.int16)-a.astype(np.int16)
            meta.append({'index':engine.dev.rows[i]['index'],'bucket':bucket,'rms_uint8':float(np.sqrt(np.mean(d.astype(float)**2))),'zero_pixel_fraction':float(np.all(d==0,2).mean()),'labeled_area':float(y.mean())})
    x=torch.stack([rgb_tensor(im) for im in images]).cuda();obs=observability(x,model.base.encoder.jnd)
    with torch.autocast('cuda',enabled=False):first=model.carrier.stem(model.carrier.filters(x,obs));f32,z32=model.carrier(x,obs)
    with torch.autocast('cuda',dtype=torch.bfloat16):first_amp=model.carrier.stem(model.carrier.filters(x,obs))
    # forward explicitly keeps first learned projection in FP32; bypass above
    # only measures the danger of accidentally removing that precision firewall.
    with torch.autocast('cuda',dtype=torch.bfloat16):fa,za=model.carrier(x,obs)
    def describe(t):
        d=t[1::2].float()-t[0::2].float();return {'paired_difference_rms':float(d.square().mean().sqrt()),'paired_exact_equal_fraction':float((d==0).float().mean())}
    result={'pixel':describe(x*255),'first_projection_FP32':describe(first),'first_projection_if_cast_BF16_NOT_USED':describe(first_amp),'local_features_FP32':describe(f32),'local_features_mixed':describe(fa),'local_logits_FP32':describe(z32),'local_logits_mixed':describe(za),'logit_amp_vs_fp32_rmse':float((za.float()-z32).square().mean().sqrt()),'per_pair':meta,'local_pixels_dtype':str(x.dtype)}
    assert result['pixel']['paired_difference_rms']>0 and result['first_projection_FP32']['paired_difference_rms']>0
    assert result['local_features_mixed']['paired_difference_rms']>0,'Signal vanished completely in AMP'
    atomic_json(path,result);print('FIDELITY',result,flush=True)
    return result
