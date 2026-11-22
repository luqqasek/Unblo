"""Losslessly compress a PyTorch ZIP archive and verify the exact submission tensors."""
import argparse
import copy
import json
from pathlib import Path
import zipfile

import torch

from experiments.native_localizer.common import atomic_json, sha256
from watermark_localization.submission import validate_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    assert not a.destination.exists()
    names = [r['image_name'] for r in json.loads((a.data / 'test/metadata.json').read_text())]
    before = torch.load(a.source, map_location='cpu', weights_only=True)
    validate_submission(before, names)
    with zipfile.ZipFile(a.source) as zin, zipfile.ZipFile(a.destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            compressed = copy.copy(item)
            compressed.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(compressed, data)
    after = torch.load(a.destination, map_location='cpu', weights_only=True)
    validate_submission(after, names)
    assert before.keys() == after.keys()
    assert before['image_names'] == after['image_names']
    assert before['masks'].dtype == after['masks'].dtype
    assert torch.equal(before['masks'], after['masks'])
    assert a.destination.stat().st_size <= 20_000_000
    report = {'source_sha256': sha256(a.source), 'destination_sha256': sha256(a.destination),
              'source_bytes': a.source.stat().st_size, 'destination_bytes': a.destination.stat().st_size,
              'tensor_values_and_dtype_identical': True, 'image_names_identical': True,
              'torch_load_verified': torch.__version__, 'compression_only': True}
    atomic_json(a.destination.with_suffix('.validation.json'), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
