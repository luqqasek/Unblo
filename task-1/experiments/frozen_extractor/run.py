"""Experiment B: frozen official extractors, independent lightweight FPN heads."""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments/baseline')]
from run import partition
from wmloc.data import LocalisationDataset
from wmloc.utils import score, miou_image, to_tensor
from wmloc.watermarks import load_watermark

OUT = Path(__file__).parent / 'results'
CONFIG = dict(seed=42, batch_size=8, max_epochs=50, patience=8, lr=.001,
              weight_decay=.0001, channels=32, fusion_size=32, bce=.5, dice=.5,
              augmentation=False, threshold_candidates=[.3,.4,.5,.6,.7], device='cpu')


class Head(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(c,32,1),nn.GroupNorm(8,32),nn.ReLU()) for c in channels])
        self.fuse = nn.Sequential(nn.Conv2d(32*len(channels),32,3,padding=1),nn.GroupNorm(8,32),nn.ReLU(),
                                  nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,1,1))
    def forward(self, features):
        x = torch.cat([F.interpolate(p(f),size=(32,32),mode='bilinear',align_corners=False)
                       for p,f in zip(self.projections,features)],1)
        return F.interpolate(self.fuse(x),size=(128,128),mode='bilinear',align_corners=False)[:,0]


def audit(ds):
    fit,val = partition(ds.meta)
    hashes, thumbs = [], []
    for item in ds:
        im = item['image']
        hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
        gray = np.asarray(Image.fromarray(im).convert('L').resize((17,16)), dtype=float)
        thumbs.append((gray[:,1:] > gray[:,:-1]).ravel())
    duplicate_pairs, similar = [], []
    thumbs = np.stack(thumbs)
    for i in fit:
        for j in val:
            if hashes[i] == hashes[j]: duplicate_pairs.append([ds.meta[i]['image_name'],ds.meta[j]['image_name']])
            if np.count_nonzero(thumbs[i] != thumbs[j]) <= 12:
                similar.append([ds.meta[i]['image_name'],ds.meta[j]['image_name']])
    result = dict(fit_names=[ds.meta[i]['image_name'] for i in fit], validation_names=[ds.meta[i]['image_name'] for i in val],
                  exact_cross_split_duplicates=duplicate_pairs, dhash_cross_split_candidates=similar,
                  metadata_keys=sorted(ds.meta[0]), note='dHash is only a heuristic; source identity not guaranteed')
    (OUT/'split_and_duplicates.json').write_text(json.dumps(result,indent=2))
    if duplicate_pairs: raise RuntimeError('Exact duplicate leakage requires grouped split')
    return fit,val


def extract(method, ds, *, cache_dir=None, spatial_limit=32):
    destination = OUT if cache_dir is None else Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path=destination/f'{method}_features.pt'
    if path.exists(): return torch.load(path,weights_only=True)
    start=time.perf_counter()
    wm=load_watermark(method,models_dir=ROOT/'checkpoints',device='cpu')
    collected={}; handles=[]
    if method=='pixelseal':
        wm.model.requires_grad_(False)
        layers={f'stage{i}':stage for i,stage in enumerate(wm.det.convnext.stages)}
    elif method=='trustmark':
        wm.net.requires_grad_(False)
        layers={f'layer{i}':getattr(wm.net,f'layer{i}') for i in range(1,5)}
    else:
        import onnx, onnxruntime as ort
        model=onnx.load(ROOT/'checkpoints/stegastamp/stega_stamp_logits.onnx')
        names=[f'gen_decoder/dec_conv_{i}/Relu:0' for i in (2,4,6,7)]
        for name in names: model.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None))
        derived=destination/'stegastamp_features.onnx'; onnx.save(model,derived)
        options=ort.SessionOptions(); options.intra_op_num_threads=4
        session=ort.InferenceSession(str(derived),options,providers=['CPUExecutionProvider'])
        layers={}
    for name,layer in layers.items():
        handles.append(layer.register_forward_hook(lambda m,a,o,key=name: collected.__setitem__(key,o.detach())))
    xs=[]; bits=[]; image_names=[]; raw_shapes=None
    with torch.inference_mode():
        for i,item in enumerate(ds):
            if item['method']!=method: continue
            image=to_tensor(item['image'])
            if method=='stegastamp':
                values=session.run(names+[wm.LOGITS],{'image':wm._prep(image),'secret':np.zeros((1,100),np.float32)})
                features=[torch.from_numpy(v) for v in values[:-1]]; logits=torch.from_numpy(values[-1])
                if not xs: assert torch.allclose(logits,wm.logits(image),atol=1e-5)
            else:
                logits=wm.logits(image); features=[collected[n] for n in layers]
            if raw_shapes is None: raw_shapes=[list(f.shape) for f in features]
            assert all(f.ndim==4 and torch.isfinite(f).all() for f in features)
            # Default cap is 32; higher-detail experiments use a separate cache.
            xs.append([F.adaptive_avg_pool2d(f,(min(spatial_limit,f.shape[-2]),min(spatial_limit,f.shape[-1])))[0].clone() for f in features])
            msg=torch.from_numpy(item['message'])
            assert logits.shape==(1,len(msg))
            bits.append(float(((logits[0]>0)==msg.bool()).float().mean()))
            image_names.append(item['image_name'])
            if len(xs)%25==0: print(method,'features',len(xs),flush=True)
    for h in handles: h.remove()
    features=[torch.stack([x[k] for x in xs]) for k in range(len(xs[0]))]
    result=dict(features=features,names=image_names,bit_accuracy=bits,raw_shapes=raw_shapes,
                layers=list(layers) if layers else names,seconds=time.perf_counter()-start)
    torch.save(result,path)
    return result


def metrics(prob,gt,threshold):
    pred=prob>=threshold
    values=[miou_image(p,g) for p,g in zip(pred,gt)]
    classes=[]
    for c in (0,1):
        union=((pred==c)|(gt==c)).sum((1,2)); inter=((pred==c)&(gt==c)).sum((1,2))
        classes.append(float(np.divide(inter,union,out=np.ones_like(union,dtype=float),where=union!=0).mean()))
    return dict(miou=score(pred,gt),iou_clean=classes[0],iou_watermark=classes[1],median=float(np.median(values)),
                std=float(np.std(values)),per_image=values,threshold=threshold)


def train(method,cache,ds,fit,val):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    names=cache['names']; lookup={n:i for i,n in enumerate(names)}
    train_ids=[lookup[ds.meta[i]['image_name']] for i in fit if ds.meta[i]['method']==method]
    val_ids=[lookup[ds.meta[i]['image_name']] for i in val if ds.meta[i]['method']==method]
    y=torch.from_numpy(np.stack([ds.m128[n] for n in names])).float()
    features=cache['features']; head=Head([f.shape[1] for f in features])
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    best=-1; stale=0; history=[]; start=time.perf_counter()
    def infer(ids):
        head.eval()
        with torch.inference_mode():
            return torch.cat([head([f[ids[j:j+8]] for f in features]).sigmoid() for j in range(0,len(ids),8)]).numpy()
    for epoch in range(50):
        head.train(); losses=[]
        for ids in np.array(train_ids)[np.random.permutation(len(train_ids))].reshape(-1,8):
            optimizer.zero_grad(); logits=head([f[ids] for f in features]); target=y[ids]
            p=logits.sigmoid(); dice=1-((2*(p*target).sum((1,2))+1)/(p.sum((1,2))+target.sum((1,2))+1)).mean()
            loss=.5*F.binary_cross_entropy_with_logits(logits,target)+.5*dice
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        prob=infer(val_ids); measured=metrics(prob,y[val_ids].numpy(),.5)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),miou=measured['miou']))
        print(method,'epoch',epoch+1,'mIoU',round(measured['miou'],5),flush=True)
        if measured['miou']>best:
            best=measured['miou']; stale=0
            torch.save(dict(state_dict=head.state_dict(),channels=[f.shape[1] for f in features],epoch=epoch+1,config=CONFIG),OUT/f'{method}_head.pt')
        else: stale+=1
        if stale>=8: break
    checkpoint=torch.load(OUT/f'{method}_head.pt',weights_only=True); head.load_state_dict(checkpoint['state_dict'])
    t=time.perf_counter(); prob=infer(val_ids); head_time=(time.perf_counter()-t)/len(val_ids)
    threshold_scores=[metrics(prob,y[val_ids].numpy(),t) for t in CONFIG['threshold_candidates']]
    selected=max(threshold_scores,key=lambda m:m['miou'])
    bit=np.asarray(cache['bit_accuracy'])[val_ids]; vals=selected['per_image']
    report=dict(method=method,config=CONFIG,layers=cache['layers'],raw_shapes=cache['raw_shapes'],cache_shapes=[list(f.shape) for f in features],
                epoch=checkpoint['epoch'],history=history,selected=selected,threshold_scores=threshold_scores,
                val_names=[names[i] for i in val_ids],bit_accuracy=bit.tolist(),mean_bit_accuracy=float(bit.mean()),
                bit_miou_pearson=float(np.corrcoef(bit,vals)[0,1]) if np.std(bit)>0 and np.std(vals)>0 else None,
                feature_seconds_per_image=cache['seconds']/170,head_seconds_per_image=head_time,training_seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in head.parameters()))
    np.savez_compressed(OUT/f'{method}_validation.npz',probability=prob,target=y[val_ids].numpy(),names=report['val_names'])
    (OUT/f'{method}_metrics.json').write_text(json.dumps(report,indent=2))
    visualize(method,report,prob,y[val_ids].numpy())


def visualize(method,report,prob,gt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ranks=np.argsort(report['selected']['per_image']); chosen=ranks[np.linspace(0,33,10).astype(int)]
    for i in chosen:
        image=np.asarray(Image.open(ROOT/'data/train/images'/f"{report['val_names'][i]}.png").convert('RGB'))
        rgb=np.asarray(Image.fromarray(image).resize((128,128)))/255
        mask=prob[i]>=report['selected']['threshold']; overlay=rgb.copy(); overlay[mask]=.5*overlay[mask]+.5*np.array([0,1,0])
        errors=np.zeros((128,128,3)); errors[mask & (gt[i]==0)]=[1,0,0]; errors[~mask & (gt[i]==1)]=[0,0,1]
        fig,axs=plt.subplots(1,6,figsize=(18,3))
        for ax,im,title in zip(axs,[image,gt[i],prob[i],mask,overlay,errors],['RGB','GT','Probability','Mask','Overlay','FP red / FN blue']):
            ax.imshow(im,vmin=0,vmax=1,cmap='viridis' if title=='Probability' else 'gray'); ax.set_title(title); ax.axis('off')
        fig.suptitle(f"{method} {report['val_names'][i]} mIoU={report['selected']['per_image'][i]:.4f}")
        fig.tight_layout(); fig.savefig(OUT/f"{method}_{i:02}_example.png"); plt.close(fig)
    fig,ax=plt.subplots(); ax.hist(report['selected']['per_image'],bins=10,range=(0,1)); ax.set(xlabel='Per-image mIoU',ylabel='Count',title=method)
    fig.savefig(OUT/f'{method}_distribution.png'); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--method',choices=['pixelseal','trustmark','stegastamp','all'],default='all'); args=ap.parse_args()
    torch.set_num_threads(4); OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'config.json').write_text(json.dumps(CONFIG,indent=2))
    ds=LocalisationDataset(ROOT/'data','train'); fit,val=audit(ds)
    for method in (['pixelseal','trustmark','stegastamp'] if args.method=='all' else [args.method]):
        cache=extract(method,ds); train(method,cache,ds,fit,val)
    if args.method=='all':
        import subprocess
        subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],check=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name('verify.py'))],check=True)


if __name__=='__main__': main()
