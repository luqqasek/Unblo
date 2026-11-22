"""Frozen recipe inference; optionally average parent and continuation probabilities."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.native_localizer.model import NativeLocalizer
from experiments.native_localizer.common import atomic_json, rgb_tensor, sha256
from experiments.next_ideas.audit import fallback
from wmloc.data import LocalisationDataset
from watermark_localization.submission import save_submission, validate_submission


def load(path, expected_step):
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['step'] == expected_step and state['config']['all_real']
    assert state['config']['real_train_n'] == 170
    model = NativeLocalizer(checkpointing=False, amp=False).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--part', type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    root = a.root / 'ss1000-screen-runs'
    recipe_path = root / 'selected_recipe.json'
    recipe = json.loads(recipe_path.read_text())
    assert recipe['recipe_frozen'] and recipe['threshold'] == .5 and not recipe['tta']
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = [load(recipe['checkpoint'], 1000)]
    if recipe['candidate'] == 'equal_probability_450_1000':
        models.append(load(recipe['parent_checkpoint'], 450))
    ds = LocalisationDataset('data', 'test', methods=['stegastamp'])
    assert len(ds) == 340 and not ds.train
    out = root / 'production'
    out.mkdir(exist_ok=True)
    dest = out / f'part{a.part}.pt'
    if dest.exists():
        receipt = json.loads(dest.with_suffix('.json').read_text())
        assert receipt['artifact_sha256'] == sha256(dest) and receipt['recipe_sha256'] == sha256(recipe_path)
        return
    indices = list(range(a.part, 340, 2))
    names, masks, probabilities = [], [], []
    start = time.monotonic()
    with torch.inference_mode():
        for begin in range(0, len(indices), 4):
            items = [ds[i] for i in indices[begin:begin+4]]
            x = torch.stack([rgb_tensor(it['image']) for it in items]).cuda()
            prob = sum(model(x)['logits'].sigmoid() for model in models) / len(models)
            for item, pp in zip(items, prob.cpu().numpy()):
                names.append(item['image_name'])
                probabilities.append(pp)
                masks.append(fallback(pp, 'top1pct').astype(np.uint8))
    save_submission(dest, names, torch.tensor(np.stack(masks)), names)
    validate_submission(torch.load(dest, weights_only=True), names)
    np.savez_compressed(out / f'part{a.part}.npz', names=names, probabilities=probabilities)
    atomic_json(dest.with_suffix('.json'), {'n': len(names), 'seconds': time.monotonic()-start,
                'artifact_sha256': sha256(dest), 'recipe_sha256': sha256(recipe_path),
                'checkpoint_sha256': sha256(recipe['checkpoint']),
                'parent_checkpoint_sha256': sha256(recipe['parent_checkpoint']),
                'recipe': recipe, 'test_GT_used': False})
    print('TEST_DONE', a.part, len(names), time.monotonic()-start, flush=True)


if __name__ == '__main__':
    main()
