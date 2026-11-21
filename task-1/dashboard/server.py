"""Local, read-only dataset explorer. Run with the task's Python environment."""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class DatasetStore:
    def __init__(self, data):
        self.data = Path(data)
        self.rows = []
        self.records = {}
        self.grids = {}
        for split in ('train', 'test'):
            folder = self.data / split
            meta = json.loads((folder / 'metadata.json').read_text())
            if split == 'train':
                with np.load(folder / 'masks_128.npz') as z:
                    self.grids = dict(zip(z['names'].tolist(), z['masks'].copy()))
            for m in meta:
                row = dict(m, split=split, bits=len(m['message']))
                row['coverage'] = float(self.grids[m['image_name']].mean()) if split == 'train' else None
                self.rows.append(row)
                self.records[(split, m['image_name'])] = row
        self.summary = {split: dict(Counter(r['method'] for r in self.rows if r['split'] == split)) for split in ('train', 'test')}

    @lru_cache(maxsize=256)
    def asset(self, split, name, kind, thumb=False):
        if (split, name) not in self.records or kind not in ('image', 'mask', 'grid'):
            raise KeyError('Unknown asset')
        if split != 'train' and kind != 'image':
            raise KeyError('Test masks are not available')
        if kind == 'grid':
            im = Image.fromarray(self.grids[name].astype(np.uint8) * 255)
        else:
            location = self.data / split / ('images' if kind == 'image' else 'masks') / (name + '.png')
            with Image.open(location) as source:
                im = source.convert('RGB' if kind == 'image' else 'L')
        if thumb:
            im.thumbnail((160, 160))
        out = BytesIO()
        im.save(out, format='PNG')
        return out.getvalue()


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == '/':
                    return self.send_data(200, (Path(__file__).parent / 'index.html').read_bytes(), 'text/html; charset=utf-8')
                if request.path == '/api/catalog':
                    return self.send_data(200, json.dumps({'rows': store.rows, 'summary': store.summary}).encode(), 'application/json')
                if request.path == '/asset':
                    q = parse_qs(request.query)
                    data = store.asset(q['split'][0], q['name'][0], q['kind'][0], q.get('thumb', ['0'])[0] == '1')
                    return self.send_data(200, data, 'image/png')
                return self.send_data(404, b'Not found', 'text/plain')
            except (KeyError, FileNotFoundError, ValueError):
                return self.send_data(404, b'Asset unavailable', 'text/plain')

        def send_data(self, status, data, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    try:
        store = DatasetStore(args.data)
    except FileNotFoundError:
        parser.exit(1, 'Brak danych. Uruchom najpierw scripts/setup.ps1 (Windows) lub scripts/setup.sh.\n')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(store))
    print(f'Dashboard: http://127.0.0.1:{args.port}  |  Ctrl+C aby zakończyć', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
