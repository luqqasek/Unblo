"""Verify real dataset records, official metrics, decoder inference, and submission."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter
import numpy as np
import torch
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, to_tensor
from wmloc.watermarks import load_watermark
from watermark_localization.submission import save_submission, validate_submission

root = Path(__file__).resolve().parents[1]
train = LocalisationDataset(root / 'data', 'train')
test = LocalisationDataset(root / 'data', 'test')
assert len(train) == 510 and len(test) == 1020
for split, ds in [('train', train), ('test', test)]:
    names = set()
    for item in ds:
        assert item['image'].shape == (512, 512, 3)
        assert item['image_name'] not in names
        names.add(item['image_name'])
        if split == 'train':
            assert np.array_equal(downsample_mask(item['mask']), item['mask128'])
    print(split, len(ds), dict(Counter(m['method'] for m in ds.meta)), flush=True)
for method in ['pixelseal', 'trustmark', 'stegastamp']:
    item = next(train[i] for i, m in enumerate(train.meta) if m['method'] == method)
    model = load_watermark(method, models_dir=root / 'checkpoints', device='cpu')
    image = to_tensor(item['image'])
    logits = model.logits(image)
    assert logits.shape == (1, model.nbits) and torch.isfinite(logits).all()
    features = model.features(image)
    print(method, 'logits', tuple(logits.shape), 'features', {k: tuple(v.shape) for k,v in features.items()}, flush=True)
names = [m['image_name'] for m in test.meta]
p = save_submission(root / 'outputs' / 'baseline.pt', names, torch.zeros((len(names), 128, 128), dtype=torch.uint8), names)
validate_submission(torch.load(p, weights_only=True), names)
print('Validated baseline:', p, p.stat().st_size, 'bytes (not uploaded)', flush=True)
