"""Real encoder, clean COCO covers, spatial masks, and exactly baseline-aligned features."""
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results'
sys.path[:0] = [str(ROOT), str(HERE.parent / 'diagnostic_audit')]
from wmloc.watermarks import StegaStamp
from wmloc.utils import downsample_mask, to_tensor
from execution_audit import inverse_feature_grid, FEATURE_NAMES

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def dhash(image):
    a = np.asarray(Image.fromarray(image).convert('L').resize((17, 16)))
    return (a[:, 1:] > a[:, :-1]).ravel()

def cover_and_mask(raw, annotations, rng, shape):
    original = Image.open(io.BytesIO(raw)).convert('RGB')
    w, h = original.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    crop = (left, top, left + side, top + side)
    cover = np.asarray(original.crop(crop).resize((512, 512), Image.Resampling.BICUBIC))
    mask = None
    if shape == 'object':
        candidates = rng.permutation(len(annotations))
        for j in candidates:
            polygons = annotations[j]['segmentation']
            if not isinstance(polygons, list):
                continue
            canvas = Image.new('L', (w, h)); draw = ImageDraw.Draw(canvas)
            for polygon in polygons:
                if len(polygon) >= 6:
                    draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=1)
            candidate = np.asarray(canvas.crop(crop).resize((512, 512), Image.Resampling.NEAREST)).astype(bool)
            if .005 <= candidate.mean() <= .5:
                mask = candidate
                break
        if mask is None:
            shape = 'rectangle_fallback'
    if mask is None:
        area = np.exp(rng.uniform(np.log(.005), np.log(.5)))
        aspect = np.exp(rng.uniform(np.log(1 / 8), np.log(8)))
        mw = int(np.clip(np.sqrt(area * aspect) * 512, 8, 500))
        mh = int(np.clip(area * 512 ** 2 / mw, 8, 500))
        x = int(rng.integers(0, 513 - mw)); y = int(rng.integers(0, 513 - mh))
        canvas = Image.new('L', (512, 512)); draw = ImageDraw.Draw(canvas)
        if shape == 'ellipse':
            draw.ellipse((x, y, x + mw - 1, y + mh - 1), fill=1)
        else:
            draw.rectangle((x, y, x + mw - 1, y + mh - 1), fill=1)
        mask = np.asarray(canvas).astype(bool)
    return cover, mask, shape, list(crop)

def main():
    torch.set_num_threads(4)
    for directory in ('cache', 'images', 'masks', 'shards'):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    source = ROOT / 'checkpoints/stegastamp/stega_stamp_logits.onnx'
    config = dict(seed=20261121, train_n=2000, validation_n=200, model_sha256=digest(source),
                  code_sha256=digest(__file__), source_downloads=json.loads((OUT / 'sources/downloads.json').read_text()),
                  method='stegastamp', cache_dtype='float32', embedding='unchanged wmloc.StegaStamp.embed',
                  geometry='baseline inverse_feature_grid after adaptive pooling cap32',
                  synthetic_schedule='i%20: 0 clean,1 full,remaining odd patch/even full then restore; shapes cycle rectangle,ellipse,object',
                  post_embedding_augmentations=None)
    if (OUT / 'generation_config.json').exists():
        previous = json.loads((OUT / 'generation_config.json').read_text())
        if previous != config:
            assert not any((OUT / 'shards').glob('*.pt')), 'Changed generation config; use new output directory.'
            save(OUT / 'generation_preflight_initial_config.json', previous)
            save(OUT / 'generation_config.json', config)
    else:
        save(OUT / 'generation_config.json', config)
    with zipfile.ZipFile(OUT / 'sources/annotations_trainval2017.zip') as archive:
        annotations = json.loads(archive.read('annotations/instances_val2017.json'))
    by_id = {}
    for a in annotations['annotations']:
        by_id.setdefault(a['image_id'], []).append(a)
    records = sorted(annotations['images'], key=lambda x: x['id'])
    order = np.random.default_rng(config['seed']).permutation(len(records))
    real = [np.asarray(Image.open(p).convert('RGB')) for p in sorted((ROOT / 'data/train/images').glob('*.png'))]
    real_hash = {hashlib.sha256(a.tobytes()).hexdigest() for a in real}
    real_dhash = np.stack([dhash(a) for a in real])
    del real
    graph = onnx.load(source)
    theta_name = 'gen_stn/Reshape:0'
    for name in FEATURE_NAMES + [theta_name]:
        graph.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    derived = OUT / 'cache/features.onnx'
    if not derived.exists():
        onnx.save(graph, derived)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(derived), options, providers=['CPUExecutionProvider'])
    wm = StegaStamp.__new__(StegaStamp); wm.sess = session; wm.device = 'cpu'
    baseline = torch.load(HERE.parent / 'stegastamp_stn_alignment/results/aligned_features.pt', weights_only=True)
    max_error = 0.
    for i in (0, 84, 169):
        image = np.asarray(Image.open(ROOT / 'data/train/images' / (baseline['names'][i] + '.png')).convert('RGB'))
        values = session.run(FEATURE_NAMES + [theta_name], {'image': wm._prep(to_tensor(image)), 'secret': np.zeros((1, 100), np.float32)})
        for level, value in enumerate(values[:4]):
            f = torch.from_numpy(value)
            f = F.adaptive_avg_pool2d(f, (min(32, f.shape[-2]), min(32, f.shape[-1])))
            f = F.grid_sample(f, inverse_feature_grid(values[-1][0], level, f.shape[-1]), mode='bilinear', align_corners=False, padding_mode='border')
            expected = baseline['features'][level][i:i + 1]
            # ONNX graph instrumentation / CSV affine roundtrip can change the last float32 bits.
            torch.testing.assert_close(f, expected, atol=1e-5, rtol=1e-5)
            max_error = max(max_error, float((f - expected).abs().max()))
    del baseline
    save(OUT / 'feature_preflight.json', dict(real_examples=3, levels=4, max_absolute_error=max_error, matches_active_baseline=True))
    print('FEATURE PREFLIGHT PASSED', max_error, flush=True)
    metadata = []; skipped = []; features = []; targets = []; seen_hashes = set()
    start = time.perf_counter()
    archive = zipfile.ZipFile(OUT / 'sources/val2017.zip')
    for position in order:
        if len(metadata) >= 2200:
            break
        record = records[position]
        index = len(metadata)
        rng = np.random.default_rng(config['seed'] + index)
        raw = archive.read('val2017/' + record['file_name'])
        cover, region, shape, crop = cover_and_mask(raw, by_id.get(record['id'], []), rng, ['rectangle', 'ellipse', 'object'][index % 3])
        sha = hashlib.sha256(cover.tobytes()).hexdigest()
        distance = int(np.count_nonzero(real_dhash != dhash(cover), axis=1).min())
        if sha in real_hash or sha in seen_hashes or distance <= 12:
            skipped.append(dict(coco_id=record['id'], min_real_dhash=distance, duplicate=sha in real_hash or sha in seen_hashes))
            continue
        seen_hashes.add(sha)
        name = f'synthetic_{index:04d}'
        shard_path = OUT / 'shards' / f'{index // 100:03d}.pt'
        if shard_path.exists():
            existing = json.loads((OUT / 'shards' / f'{index // 100:03d}.json').read_text())
            row = existing[index % 100]
            assert row['cover_sha256'] == sha and row['coco_id'] == record['id']
            metadata.append(row)
            continue
        bits = rng.integers(0, 2, 100, dtype=np.uint8)
        mode = 'clean' if index % 20 == 0 else 'full' if index % 20 == 1 else 'patch' if index % 2 else 'restore'
        if mode == 'clean':
            image = cover.copy(); mask = np.zeros((512, 512), bool)
        elif mode in ('full', 'restore'):
            marked = wm.embed(cover, bits)
            mask = np.ones((512, 512), bool) if mode == 'full' else ~region
            image = np.where(mask[:, :, None], marked, cover)
        else:
            yy, xx = np.where(region)
            x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
            marked = cover.copy()
            marked[y0:y1, x0:x1] = wm.embed(cover[y0:y1, x0:x1], bits)
            mask = region
            image = np.where(mask[:, :, None], marked, cover)
        assert np.array_equal(image[~mask], cover[~mask])
        gt = downsample_mask(mask.astype(np.uint8))
        outputs = session.run(FEATURE_NAMES + [theta_name, wm.LOGITS], {'image': wm._prep(to_tensor(image)), 'secret': np.zeros((1, 100), np.float32)})
        theta = outputs[-2][0]
        sample_features = []
        for level, value in enumerate(outputs[:4]):
            f = torch.from_numpy(value)
            f = F.adaptive_avg_pool2d(f, (min(32, f.shape[-2]), min(32, f.shape[-1])))
            f = F.grid_sample(f, inverse_feature_grid(theta, level, f.shape[-1]), mode='bilinear', align_corners=False, padding_mode='border')
            assert torch.isfinite(f).all()
            sample_features.append(f[0].clone())
        features.append(sample_features); targets.append(gt)
        mse = np.mean((image.astype(float) - cover.astype(float)) ** 2)
        row = dict(image_id=name, coco_id=record['id'], source_filename=record['file_name'], source_url=record.get('coco_url'),
                   flickr_url=record.get('flickr_url'), license=record.get('license'), original_sha256=hashlib.sha256(raw).hexdigest(),
                   cover_sha256=sha, image_sha256=hashlib.sha256(image.tobytes()).hexdigest(), crop=crop,
                   min_real_dhash=distance, split='train' if index < 2000 else 'validation', mode=mode, shape=shape,
                   message=''.join(map(str, bits.tolist())), area=float(gt.mean()), bit_accuracy=float(((outputs[-1][0] > 0) == bits).mean()),
                   psnr=None if mse == 0 else float(10 * np.log10(255 ** 2 / mse)), theta=theta.tolist(),
                   watermark_pixel_change=float(np.any(image != cover, axis=2)[mask].mean()) if mask.any() else None)
        metadata.append(row)
        Image.fromarray(image).save(OUT / 'images' / f'{name}.png')
        Image.fromarray(mask.astype(np.uint8) * 255).save(OUT / 'masks' / f'{name}.png')
        if (index + 1) % 100 == 0:
            chunk = dict(names=[r['image_id'] for r in metadata[-100:]], features=[torch.stack([f[k] for f in features]) for k in range(4)],
                         target=torch.from_numpy(np.stack(targets)))
            torch.save(chunk, shard_path)
            save(shard_path.with_suffix('.json'), metadata[-100:])
            features.clear(); targets.clear()
        if (index + 1) % 20 == 0:
            print(f'GENERATED {index + 1}/2200 seconds={time.perf_counter() - start:.1f} last={mode} area={gt.mean():.4f} bits={row["bit_accuracy"]:.3f}', flush=True)
    assert len(metadata) == 2200
    save(OUT / 'manifest.json', metadata)
    save(OUT / 'source_quality.json', dict(exact_unique_covers=len(seen_hashes), rejected=skipped,
         licenses=annotations['licenses'], min_dhash_to_real=min(r['min_real_dhash'] for r in metadata),
         limitation='No source-identity metadata for competition; exact and dHash screening cannot rule out all shared-source variants.',
         seconds=time.perf_counter() - start))
    assert digest(source) == config['model_sha256']
    print('GENERATION COMPLETE', flush=True)

if __name__ == '__main__':
    main()
