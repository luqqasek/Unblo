"""Replace only selected empty PixelSeal masks, preserving #3188 elsewhere."""
import json
from pathlib import Path

import torch

from experiments.native_localizer.common import atomic_json, sha256
from watermark_localization.submission import save_submission, validate_submission


def main():
    here = Path(__file__).parent
    task = here.parents[1]
    result = here / 'results'
    evidence = json.loads((result / 'evaluation.json').read_text())
    inference = json.loads((result / 'production/completed.json').read_text())
    assert evidence['selected'] == 'local_top164'
    assert inference['evidence_sha256'] == sha256(result / 'evaluation.json')
    deployment = result / 'production/deployment_validation.json'
    assert sha256(deployment) == inference['deployment_evidence_sha256']
    assert json.loads(deployment.read_text())['passed']
    subset = result / 'production/pixelseal-subset.pt'
    assert sha256(subset) == inference['artifact_sha256']
    previous = here.parent / 'pcmv/results/base-candidate/submission/submission-ps-global1561.pt'
    assert sha256(previous) == 'd8bb4c93e7e5a45b32beae781189b47af7401c6612c3336611d4d14b13d98f5f'
    metadata = json.loads((task / 'data/test/metadata.json').read_text())
    names = [r['image_name'] for r in metadata]
    family = {r['image_name']: r['method'] for r in metadata}
    old = torch.load(previous, map_location='cpu', weights_only=True)
    validate_submission(old, names)
    oldm = dict(zip(old['image_names'], old['masks']))
    obj = torch.load(subset, map_location='cpu', weights_only=True)
    validate_submission(obj, obj['image_names'])
    replacement = dict(zip(obj['image_names'], obj['masks']))
    assert len(replacement) == 340 and all(family[n] == 'pixelseal' for n in replacement)
    masks = torch.stack([replacement.get(n, oldm[n]).to(torch.uint8) for n in names])
    changed = {f: sum(not torch.equal(m, oldm[n].to(torch.uint8)) for n, m in zip(names, masks) if family[n] == f)
               for f in ('pixelseal', 'trustmark', 'stegastamp')}
    assert changed['trustmark'] == changed['stegastamp'] == 0
    assert changed['pixelseal'] == inference['changed_n'] > 0
    for n in inference['changed_names']:
        assert oldm[n].sum().item() == 0 and replacement[n].sum().item() == 164
    out = result / 'submission'
    out.mkdir(exist_ok=True)
    dest = out / 'submission-carrier-fallback.pt'
    assert not dest.exists()
    save_submission(dest, names, masks, names)
    validate_submission(torch.load(dest, weights_only=True), names)
    assert dest.stat().st_size <= 20_000_000
    report = {'base_submission': 3188, 'base_sha256': sha256(previous), 'sha256': sha256(dest),
              'bytes': dest.stat().st_size, 'shape': list(masks.shape), 'dtype': str(masks.dtype),
              'unique_names': len(set(names)) == 1020, 'binary': True, 'changed_images': changed,
              'inference': inference, 'test_GT_used': False}
    atomic_json(out / 'assembly_validation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'inference'}, indent=2))


if __name__ == '__main__':
    main()
