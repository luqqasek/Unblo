"""Download official COCO archives via HTTPS path-style S3; preserve original bytes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time
import requests

OUT = Path(__file__).parent / 'results/sources'

def fetch(name):
    path = OUT / Path(name).name
    url = 'https://s3.amazonaws.com/images.cocodataset.org/' + name
    if not path.exists():
        temp = path.with_suffix('.part')
        with requests.get(url, stream=True, timeout=(30, 20)) as response:
            response.raise_for_status()
            expected = int(response.headers['Content-Length'])
        parts = OUT / (path.name + '.chunks'); parts.mkdir(exist_ok=True)
        size = 8 * 1024 * 1024
        def part(i):
            first, last = i * size, min((i + 1) * size, expected) - 1
            target = parts / f'{i:04d}'
            if target.exists() and target.stat().st_size == last - first + 1:
                return target
            for attempt in range(5):
                try:
                    r = requests.get(url, headers={'Range': f'bytes={first}-{last}'}, timeout=(20, 20))
                    r.raise_for_status()
                    assert r.status_code == 206 and len(r.content) == last - first + 1
                    target.write_bytes(r.content)
                    return target
                except Exception:
                    if attempt == 4:
                        raise
        with ThreadPoolExecutor(6) as pool:
            files = list(pool.map(part, range((expected + size - 1) // size)))
        with temp.open('wb') as stream:
            for file in files:
                stream.write(file.read_bytes())
        assert temp.stat().st_size == expected
        temp.replace(path)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    print('READY', path.name, flush=True)
    return dict(url=url, filename=path.name, sha256=sha, bytes=path.stat().st_size)

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(2) as pool:
        records = list(pool.map(fetch, ['zips/val2017.zip', 'annotations/annotations_trainval2017.zip']))
    (OUT / 'downloads.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
