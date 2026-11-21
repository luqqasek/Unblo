"""One explicit leaderboard upload. API key is read from stdin, never persisted."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import requests
import torch
from watermark_localization.submission import validate_submission
from .common import atomic_json,sha256

def main():
    p=argparse.ArgumentParser(); p.add_argument('--file',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--receipt',type=Path,required=True); a=p.parse_args()
    if a.receipt.exists(): raise FileExistsError('Check existing receipt before attempting another upload')
    names=[r['image_name'] for r in json.loads((a.data/'test/metadata.json').read_text())]
    obj=torch.load(a.file,map_location='cpu',weights_only=True)
    validate_submission(obj,names)
    if len(names)!=1020 or a.file.stat().st_size>20_000_000: raise ValueError('Invalid size or image count')
    key=sys.stdin.readline().strip()
    if not key: raise ValueError('Missing API key on stdin')
    url='http://34.147.229.55/submit/34-watermark-localisation-image'
    # Exactly one request: do not retry an ambiguous response and create duplicates.
    with a.file.open('rb') as f:
        resp=requests.post(url,headers={'X-API-Key':key},files={'file':(a.file.name,f,'application/octet-stream')},timeout=(10,180))
    try: body=resp.json()
    except ValueError: body={'raw_text':resp.text}
    receipt={'timestamp':datetime.now(timezone.utc).isoformat(),'endpoint':url,'artifact_sha256':sha256(a.file),'bytes':a.file.stat().st_size,'http_status':resp.status_code,'response':body}
    atomic_json(a.receipt,receipt)
    print(json.dumps(receipt,indent=2))
    resp.raise_for_status()
if __name__=='__main__':main()
