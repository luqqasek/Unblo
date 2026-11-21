"""Frozen TrustMark 0.8.0 ReMark evidence, actual API and documented map transforms."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from trustmark import TrustMark

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results'
BASE = ROOT / 'experiments/frozen_extractor/results/cv3'
spec = importlib.util.spec_from_file_location('remark_official', ROOT / 'starter-kit/task-1/wmloc/utils.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def real_data():
    z = np.load(BASE / 'trustmark_oof.npz')
    names = z['names'].tolist()
    meta = {r['image_name']: r for r in json.loads((ROOT / 'data/train/metadata.json').read_text())}
    assert len(names) == len(set(names)) == 170 and all(meta[n]['method'] == 'trustmark' for n in names)
    return names, z['target'].astype(bool), z['probability'], z['threshold'], meta


def down(x):
    a = torch.from_numpy(np.ascontiguousarray(x)).float()[None, None]
    return F.interpolate(a, size=(128, 128), mode='bilinear', align_corners=False, antialias=True)[0, 0].numpy()


def variants(maps):
    out = {}
    for name, value in maps.items():
        value = down(value) if value.shape != (128, 128) else value.astype(np.float32)
        for sigma in (0, 1, 3):
            out[f'{name}_s{sigma}'] = ndi.gaussian_filter(value, sigma, mode='reflect') if sigma else value
    return out


def image_statistics(image):
    rgb = np.asarray(image, np.float32) / 255
    gray = rgb @ np.array([.299, .587, .114], np.float32)
    mean = ndi.gaussian_filter(gray, 8, mode='reflect')
    std = np.sqrt(np.maximum(ndi.gaussian_filter(gray * gray, 8, mode='reflect') - mean * mean, 0))
    edge = np.hypot(ndi.sobel(gray, 0, mode='reflect'), ndi.sobel(gray, 1, mode='reflect')) / 8
    hf = np.abs(gray - ndi.gaussian_filter(gray, 2, mode='reflect'))
    return std, edge, hf


class Remover:
    def __init__(self, device='cuda:0'):
        assert torch.cuda.is_available(), 'GPU required for inference.'
        assert importlib.metadata.version('trustmark') == '0.8.0'
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.set_num_threads(4)
        # 0.8.0 always loads removal; loadRemover is not in this pinned version's signature.
        self.tm = TrustMark(model_type='Q', use_ECC=False, secret_len=100, verbose=False, device=device)
        for net in (self.tm.encoder, self.tm.decoder, self.tm.removal):
            net.eval().requires_grad_(False)
        net = self.tm.removal.denoise
        self.observed, self.handles, self.capture = {}, [], True
        layers = {'R5_enc0': net.encoders[0], 'R5_enc2': net.encoders[2], 'R5_middle': net.middle_blks, 'R5_dec3': net.decoders[3]}
        for name, layer in layers.items():
            def hook(module, inputs, output, key=name):
                if self.capture:
                    self.observed[key] = output.detach().float().square().mean(1).sqrt()[0].cpu().numpy()
                    self.observed[key + '_shape'] = list(output.shape)
            self.handles.append(layer.register_forward_hook(hook))
        def capture_io(module, inputs, output):
            if self.capture:
                self.observed['input'] = inputs[0].detach()
                self.observed['output'] = output.detach()
        self.handles.append(self.tm.removal.register_forward_hook(capture_io))

    @torch.inference_mode()
    def decode(self, images):
        a = torch.stack([torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2, 0, 1) for im in images]).float().to(self.device) / 255
        a = F.interpolate(a, (245, 245), mode='bilinear', align_corners=False, antialias=True) * 2 - 1
        return self.tm.decoder.decoder.decoder(a).float().cpu().numpy()

    @torch.inference_mode()
    def full(self, image):
        assert image.mode == 'RGB' and image.size == (512, 512)
        self.capture = True
        self.observed = {}
        recovered = self.tm.remove_watermark(image)
        assert recovered.mode == 'RGB' and recovered.size == image.size
        original = np.asarray(image, np.float32) / 255
        residual = original - np.asarray(recovered, np.float32) / 255
        magnitude = np.abs(residual).mean(2)
        std, edge, hf = image_statistics(image)
        low = ndi.gaussian_filter(residual, (2, 2, 0), mode='reflect')
        correction = (self.observed['output'].clamp(-1, 1) - self.observed['input']) / 2
        float_mag = correction.abs().mean(1)[0].cpu().numpy()
        maps = dict(R1=magnitude, R2=np.linalg.norm(residual, axis=2), R3=np.abs(residual - low).mean(2),
                    R4_contrast=magnitude / np.maximum(std, 1 / 255),
                    R4_local=magnitude / (ndi.gaussian_filter(magnitude, 8, mode='reflect') + 1 / 255),
                    R1_float=float_mag)
        for k, v in self.observed.items():
            if k.startswith('R5_') and not k.endswith('_shape'): maps[k] = v
        blur = ndi.gaussian_filter(original, (2, 2, 0), mode='reflect')
        roundtrip = image.resize((256, 256), Image.Resampling.BILINEAR).resize((512, 512), Image.Resampling.BILINEAR)
        maps.update(control_blur=np.abs(original - blur).mean(2),
                    control_resize=np.abs(original - np.asarray(roundtrip, np.float32) / 255).mean(2),
                    control_edges=edge)
        info = dict(input_shape=list(original.shape), output_shape=list(np.asarray(recovered).shape),
                    output_min=int(np.asarray(recovered).min()), output_max=int(np.asarray(recovered).max()),
                    input_tensor_range=[float(self.observed['input'].min()), float(self.observed['input'].max())],
                    output_tensor_range=[float(self.observed['output'].min()), float(self.observed['output'].max())],
                    feature_shapes={k: v for k, v in self.observed.items() if k.endswith('_shape')})
        return recovered, variants(maps), info

    @torch.inference_mode()
    def patches(self, patches, batch_size=4):
        """Batched exact square-crop API path; no GT enters window inference."""
        self.capture = False
        result = []
        for first in range(0, len(patches), batch_size):
            batch = patches[first:first + batch_size]
            small = [np.asarray(im.resize((256, 256), Image.Resampling.BILINEAR)).copy() for im in batch]
            x = torch.from_numpy(np.stack(small)).permute(0, 3, 1, 2).float().to(self.device) / 255 * 2 - 1
            out = self.tm.removal(x).clamp(-1, 1)
            delta = out - x
            for i, image in enumerate(batch):
                w, h = image.size
                d = F.interpolate(delta[i:i + 1], (h, w), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
                orig = np.asarray(image)
                decoded = np.clip(d + orig / 127.5 - 1, -1, 1) * 127.5 + 127.5
                recovered = self.tm.put_the_image_after_processing(decoded, orig.copy())
                result.append(Image.fromarray(recovered.astype(np.uint8)))
        return result

    def local(self, image, sizes=(128, 256, 384)):
        arrays = {}
        for size in sizes:
            positions = list(range(0, 513 - size, size // 2))
            if positions[-1] != 512 - size: positions.append(512 - size)
            boxes = [(x, y, x + size, y + size) for y in positions for x in positions]
            crops = [image.crop(b) for b in boxes]
            recovered = self.patches(crops)
            weight = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
            weight = np.maximum(weight, .01)
            value = np.zeros((512, 512), np.float32)
            denominator = np.zeros_like(value)
            for box, crop, rec in zip(boxes, crops, recovered):
                x, y, _, _ = box
                residual = np.abs(np.asarray(crop, np.float32) - np.asarray(rec, np.float32)).mean(2) / 255
                value[y:y + size, x:x + size] += weight * residual
                denominator[y:y + size, x:x + size] += weight
            assert (denominator > 0).all()
            arrays[f'local{size}'] = value / denominator
        arrays['local_mean'] = np.mean(list(arrays.values()), axis=0)
        return variants(arrays)
