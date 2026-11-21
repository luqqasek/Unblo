"""Frozen known-message evidence and label-free threshold selection; resumable."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results'
sys.path.insert(0, str(ROOT))
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def dense(wm, x, resolution):
    x = F.interpolate(x, (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
    if wm.name == 'pixelseal':
        emb = wm.det.pixel_decoder.output_upscaling(wm.det.convnext(wm.det.preprocess(x)))
        layer = wm.det.pixel_decoder.linear
        return F.conv2d(emb, layer.weight[:, :, None, None], layer.bias)[:, 1:]
    net = wm.net
    x = x * 2 - 1
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
        x = layer(x)
    return F.conv2d(x, net.fc.weight[:, :, None, None], net.fc.bias)


def global_logits(wm, x):
    if wm.name == 'pixelseal':
        return wm.det(wm._resize(x))[:, 1:]
    return wm.net(wm._prep(x))


def resize_maps(x):
    return F.interpolate(x, (128, 128), mode='bilinear', align_corners=False)


def map_statistics(z, signs):
    # One image; signs contains the true message and two independent nulls.
    signed = z * signs[:, :, None, None]
    dot = signed.mean(1)
    norm = z.square().mean(1).sqrt().clamp_min(1e-8)
    return dict(dot=dot, cos=dot/norm, acc=signed.sigmoid().mean(1), norm=norm)


def integrated(wm, x, blur, signs, steps):
    alpha = torch.arange(1, steps+1, device=x.device).float()/steps
    path = (blur + alpha[:, None, None, None]*(x-blur)).detach().requires_grad_()
    # Split paths into batches to bound GPU memory at 16 quadrature points.
    accumulated = []
    for start in range(0, steps, 4):
        part = path[start:start+4]
        z = global_logits(wm, part)
        grads = []
        for j, sign in enumerate(signs):
            g = torch.autograd.grad((z*sign).mean(1).sum(), part,
                                    retain_graph=j+1 < len(signs))[0]
            grads.append(g.detach().sum(0, keepdim=True))
        accumulated.append(torch.cat(grads))
    attribution = sum(accumulated)/steps*(x-blur)
    pixel = attribution.sum(1)
    maps = F.avg_pool2d(pixel[:, None], 4)[:, 0]
    return maps, pixel.sum((1, 2))


def incumbent(method):
    variant = {'pixelseal':'n4k_s2500', 'trustmark':'n4k_s5000'}[method]
    base = ROOT/'experiments/synthetic_scaling/results/runs'/method
    result = {}
    for fold in (1,2,3):
        zs = [np.load(base/f'{variant}_fold{fold}_seed{s}.npz') for s in (42,7,123)]
        assert all(np.array_equal(z['names'], zs[0]['names']) for z in zs)
        p = np.mean([z['probability'] for z in zs], axis=0)
        threshold = float(np.mean([z['threshold'] for z in zs]))
        for name, probability, target in zip(zs[0]['names'], p, zs[0]['target']):
            result[str(name)] = (probability, target, threshold, fold)
    return result


def select_threshold(scores, area, base_index):
    # scores order: original, blurred, keep candidates, drop candidates.
    n = len(area)
    keep, drop = scores[2:2+n], scores[2+n:]
    contrast = int(np.argmax(keep-drop))
    delta = scores[0]-scores[1]
    selected = base_index
    if delta > 0:
        eligible = np.flatnonzero((keep-scores[1] >= .95*delta) & (drop-scores[1] <= .05*delta))
        if len(eligible):
            selected = int(eligible[np.argmin(area[eligible])])
    return contrast, selected


def keep_drop(wm, x, blur, signs, p, y, threshold):
    thresholds = np.unique(np.r_[np.arange(1,10)/10, threshold])
    base_index = int(np.argmin(abs(thresholds-threshold)))
    masks = torch.from_numpy(p[None] >= thresholds[:, None, None]).float().to(x.device)
    full = F.interpolate(masks[:, None], (512,512), mode='nearest')
    inputs = torch.cat([x, blur, full*x+(1-full)*blur, (1-full)*x+full*blur])
    with torch.no_grad():
        logits = torch.cat([global_logits(wm, inputs[j:j+4]) for j in range(0,len(inputs),4)])
        scores = (logits[:, None]*signs[None]).mean(2).cpu().numpy().T
    area = masks.mean((1,2)).cpu().numpy()
    miou = np.array([miou_image(p >= t, y) for t in thresholds])
    result = {'baseline':float(miou[base_index]), 'oracle':float(miou.max()),
              'thresholds':thresholds.tolist(), 'scores':scores.tolist(),
              'baseline_threshold':threshold, 'original_bit_accuracy':float(((logits[0]*signs[0])>0).float().mean())}
    for j, label in enumerate(('true','null1','null2')):
        contrast, retain = select_threshold(scores[j], area, base_index)
        for name, index in [('contrast',contrast),('retain95',retain)]:
            result[f'{name}_{label}'] = float(miou[index])
            result[f'{name}_{label}_threshold'] = float(thresholds[index])
            result[f'{name}_{label}_area'] = float(area[index])
    return result


def metrics(maps, y):
    target = y.reshape(-1).astype(bool)
    assert target.any() and not target.all()
    return {name:dict(auc=float(roc_auc_score(target, value.ravel())),
                      ap=float(average_precision_score(target, value.ravel())))
            for name,value in maps.items()}


def process(method, limit):
    start = time.monotonic()
    wm = load_watermark(method, ROOT/'checkpoints', 'cuda:0')
    net = wm.det if method=='pixelseal' else wm.net
    net.eval().requires_grad_(False)
    records = incumbent(method)
    metadata = {m['image_name']:m for m in json.loads((ROOT/'data/train/metadata.json').read_text())}
    for index, name in enumerate(sorted(records)[:limit]):
        dest = OUT/method/f'{name}.json'
        if dest.exists():
            continue
        p,y,threshold,fold = records[name]
        meta = metadata[name]
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        bits = np.array([int(c) for c in meta['message']])
        signs = torch.tensor(2*np.stack([bits, rng.integers(0,2,len(bits)), rng.integers(0,2,len(bits))])-1,
                             device='cuda:0', dtype=torch.float32)
        rgb = np.array(Image.open(ROOT/'data/train/images'/f'{name}.png').convert('RGB'))
        x = torch.from_numpy(rgb).permute(2,0,1)[None].float().to('cuda:0')/255
        blur = gaussian_blur(x, [19,19], [3.,3.])
        maps = {}; verification = {}
        with torch.no_grad():
            native = global_logits(wm,x)
            official = wm.logits(x)
            torch.testing.assert_close(native, official, atol=1e-4, rtol=1e-4)
            for scale in (wm.resolution,512):
                z = dense(wm,x,scale)
                if scale==wm.resolution:
                    error = float((z.mean((2,3))-native).abs().max())
                    verification['projection_max_error'] = error
                    torch.testing.assert_close(z.mean((2,3)), native, atol=1e-4, rtol=1e-4)
                stats = map_statistics(z,signs)
                for stat, value in stats.items():
                    up = resize_maps(value[:,None])[:,0].cpu().numpy()
                    key = f'dense{scale}_{stat}'
                    if stat=='norm':
                        maps[key] = up[0]
                    else:
                        maps[key] = up[0]
                        maps[key+'_null'] = up[1:].mean(0)
                        maps[key+'_calibrated'] = up[0]-up[1:].mean(0)
            change = ((native-global_logits(wm,blur))*signs).mean(1)
        ig,sums = integrated(wm,x,blur,signs,4)
        ig = ig.detach().cpu().numpy()
        maps['blur_ig'] = ig[0]; maps['blur_ig_null'] = ig[1:].mean(0)
        maps['blur_ig_calibrated'] = ig[0]-ig[1:].mean(0)
        verification['ig4_sum'] = sums.tolist()
        verification['score_change'] = change.tolist()
        if index<8:
            ig16,sums16 = integrated(wm,x,blur,signs,16)
            verification['ig16_sum'] = sums16.tolist()
            verification['ig4_ig16_correlation'] = float(np.corrcoef(ig[0].ravel(),ig16[0].cpu().numpy().ravel())[0,1])
            verification['ig16_metrics'] = metrics({'true':ig16[0].cpu().numpy()},y)
        for stat in ('dot','cos','acc'):
            for suffix in ('','_null','_calibrated'):
                maps[f'multiscale_{stat}{suffix}'] = (maps[f'dense{wm.resolution}_{stat}{suffix}']+maps[f'dense512_{stat}{suffix}'])/2
        maps['baseline_probability'] = p
        assert all(np.isfinite(v).all() for v in maps.values())
        kd = keep_drop(wm,x,blur,signs,p,y,threshold)
        row = dict(name=name,method=method,fold=fold,fg_fraction=float(y.mean()),
                   full_watermark=bool(meta['full_watermark']), metrics=metrics(maps,y),
                   keep_drop=kd, verification=verification)
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest.with_suffix('.npz'),**maps)
        save(dest,row)
        if (index+1)%10==0 or index==0:
            print(method,index+1,'/',min(limit,len(records)),'seconds',round(time.monotonic()-start,1),flush=True)
    del wm, net
    torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--methods',nargs='+',default=['pixelseal','trustmark'])
    parser.add_argument('--limit',type=int,default=170)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for method in args.methods:
        process(method,args.limit)


if __name__=='__main__':
    main()
