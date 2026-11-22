"""Replace only StegaStamp in the strongest accepted #3188 artifact."""
import json
from pathlib import Path

import torch

from experiments.native_localizer.common import atomic_json, sha256
from watermark_localization.submission import save_submission, validate_submission


def main():
    here = Path(__file__).parent
    task = here.parents[1]
    result = here / 'results'
    recipe = json.loads((result / 'selected_recipe.json').read_text())
    evidence = json.loads((result / 'evaluation.json').read_text())
    assert evidence['selected'] == recipe['candidate'] and evidence['selected']
    assert recipe['evidence_sha256'] == sha256(result / 'evaluation.json')
    previous = here.parent / 'carrier_fallback/results/submission/submission-carrier-fallback-compressed.pt'
    base_receipt = json.loads(previous.with_name('server_receipt.json').read_text())
    assert base_receipt['response']['submission_id'] == 3457 and base_receipt['http_status'] == 200
    assert sha256(previous) == base_receipt['artifact_sha256'] == '1d588209f956a5fb86fbbb7cccb65dd7f095bcf2486a625756e11c53891f4b65'
    metadata = json.loads((task / 'data/test/metadata.json').read_text())
    names = [r['image_name'] for r in metadata]
    family = {r['image_name']: r['method'] for r in metadata}
    old = torch.load(previous, map_location='cpu', weights_only=True)
    validate_submission(old, names)
    oldm = dict(zip(old['image_names'], old['masks']))
    replacement, receipts = {}, []
    for part in range(2):
        path = result / f'production/part{part}.pt'
        obj = torch.load(path, map_location='cpu', weights_only=True)
        receipt = json.loads(path.with_suffix('.json').read_text())
        validate_submission(obj, obj['image_names'])
        assert receipt['artifact_sha256'] == sha256(path)
        assert receipt['recipe_sha256'] == sha256(result / 'selected_recipe.json')
        assert receipt['n'] == len(obj['image_names']) == 170
        receipts.append(receipt)
        for n, mask in zip(obj['image_names'], obj['masks']):
            assert n not in replacement and family[n] == 'stegastamp'
            replacement[n] = mask
    assert len(replacement) == 340
    assert len({r['checkpoint_sha256'] for r in receipts}) == 1
    masks = torch.stack([replacement.get(n, oldm[n]).to(torch.uint8) for n in names])
    changed = {f: sum(not torch.equal(m, oldm[n].to(torch.uint8)) for n, m in zip(names, masks) if family[n] == f)
               for f in ('pixelseal', 'trustmark', 'stegastamp')}
    assert changed['pixelseal'] == changed['trustmark'] == 0 and changed['stegastamp'] > 0
    out = result / 'submission'
    out.mkdir(exist_ok=True)
    dest = out / 'submission-ss1000.pt'
    assert not dest.exists()
    save_submission(dest, names, masks, names)
    validate_submission(torch.load(dest, weights_only=True), names)
    assert dest.stat().st_size <= 20_000_000
    report = {'base_submission': 3457, 'base_sha256': sha256(previous), 'sha256': sha256(dest),
              'bytes': dest.stat().st_size, 'shape': list(masks.shape), 'dtype': str(masks.dtype),
              'unique_names': len(set(names)) == 1020, 'binary': True, 'changed_images': changed,
              'recipe': recipe, 'sources': receipts, 'test_GT_used': False}
    atomic_json(out / 'assembly_validation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('recipe', 'sources')}, indent=2))


if __name__ == '__main__':
    main()
