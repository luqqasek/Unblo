"""Matched clean/marked FP32 versus deeper AMP across local-crop scales/remnants."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.utils import downsample_mask
from .common import atomic_json, rgb_tensor, seed_all
from .generator import EncodedDataset, rect_mask
from .losses import segmentation
from .model import NativeLocalizer
from .profile import delta_stats


def cosine(a,b):
    return float(F.cosine_similarity(a.float().flatten()[None],b.float().flatten()[None],dim=1)[0])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--backbone",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--count",type=int,default=16)
    a=p.parse_args(); seed_all(42); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    data=EncodedDataset(a.manifest)
    model=NativeLocalizer(backbone_path=a.backbone,checkpointing=False).to(a.device).eval()
    order=sorted(range(len(data.rows)),key=lambda i:np.prod(data.rows[i]["crop_resolution"]))
    selected=sorted(set(order[int(i)] for i in np.linspace(0,len(order)-1,min(a.count,len(order)))))
    rows=[]
    for index in selected:
        clean,marked,meta=data.load(index)
        images=torch.stack([rgb_tensor(clean),rgb_tensor(marked)]).to(a.device)
        target=torch.from_numpy(np.stack([np.zeros((128,128),np.uint8),downsample_mask(rect_mask(meta["rectangle"]))])).to(a.device)
        gradients={}
        outputs={}
        losses={}
        for amp in (False,True):
            model.amp=amp; model.zero_grad(set_to_none=True)
            output=model(images,return_features=True)
            loss=segmentation(output["logits"],target); loss.backward()
            keys=["encoder.stem.0.weight","encoder.stages.0.blocks.0.conv_dw.weight",
                  "encoder.stages.1.blocks.0.conv_dw.weight","encoder.stages.2.blocks.0.conv_dw.weight",
                  "encoder.stages.3.blocks.0.conv_dw.weight","local.first.0.weight","fuse.1.weight"]
            gradients[amp]={k:v.grad.detach().clone() for k,v in model.named_parameters() if k in keys}
            outputs[amp]={k:output[k].detach() for k in ("logits","first","local")}
            losses[amp]=float(loss.detach())
            del output,loss
        fp,mp=outputs[False],outputs[True]
        d_fp=fp["logits"][1]-fp["logits"][0]; d_mp=mp["logits"][1]-mp["logits"][0]
        rows.append({
            "realization_id":meta["realization_id"],"regime":meta["regime"],"crop_resolution":meta["crop_resolution"],
            "raw_rgb_delta":delta_stats(images[1],images[0]),
            "first_projection_precision_error":delta_stats(fp["first"],mp["first"]),
            "local_stem_precision_error":delta_stats(fp["local"],mp["local"]),
            "logit_precision_error":delta_stats(fp["logits"],mp["logits"]),
            "paired_response_fp32":delta_stats(fp["logits"][1],fp["logits"][0]),
            "paired_response_amp":delta_stats(mp["logits"][1],mp["logits"][0]),
            "paired_response_cosine":cosine(d_fp,d_mp),"loss_fp32":losses[False],"loss_amp":losses[True],
            "gradient_cosines":{k:cosine(gradients[False][k],gradients[True][k]) for k in gradients[False]},
            "all_gradients_finite":all(bool(torch.isfinite(v).all()) for g in gradients.values() for v in g.values()),
        })
    result={"n_pairs":len(rows),"rows":rows,
            "first_projection_exact":all(r["first_projection_precision_error"]["max_abs"]==0 for r in rows),
            "local_stem_exact":all(r["local_stem_precision_error"]["max_abs"]==0 for r in rows),
            "paired_response_cosine_min":min(r["paired_response_cosine"] for r in rows),
            "paired_response_cosine_mean":float(np.mean([r["paired_response_cosine"] for r in rows])),
            "all_gradients_finite":all(r["all_gradients_finite"] for r in rows),
            "training_amp_decision":"pending review; this test does not enable AMP itself"}
    atomic_json(a.out,result); print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
