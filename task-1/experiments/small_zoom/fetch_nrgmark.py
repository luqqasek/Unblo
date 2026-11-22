"""Fetch pinned, isolated detector source/weights without upgrading TrustMark."""
import argparse,hashlib,io,json,urllib.request,zipfile
from pathlib import Path
COMMIT='4c60b9fdb4125fca3001f0803e6890a6b55371ac'
SHA='7e1da5667a4da070573d71bbf5a03d586f8836397bbcdf2c2044865ec37df0a5'
URL='https://cai-watermark.adobe.net/watermarking/trustmark-models/trustmark_bbox_Q.ckpt'
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'checkpoints/nrgmark';out.mkdir(parents=True,exist_ok=True)
    model=out/'trustmark_bbox_Q.ckpt'
    if not model.exists():
        temp=model.with_suffix('.part');urllib.request.urlretrieve(URL,temp);assert hashlib.sha256(temp.read_bytes()).hexdigest()==SHA;temp.replace(model)
    assert hashlib.sha256(model.read_bytes()).hexdigest()==SHA
    url=f'https://codeload.github.com/adobe/trustmark/zip/{COMMIT}'
    archive=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read()));prefix=f'trustmark-{COMMIT}/python/trustmark/';hashes={}
    for name in archive.namelist():
        if not name.startswith(prefix) or name.endswith('/'):continue
        rel=name[len(prefix):]
        if not (rel=='yolo_wo_decoder_multibox.py' or rel.startswith('bbox_detector/') and rel.endswith('.py')):continue
        path=out/'source'/rel;path.parent.mkdir(parents=True,exist_ok=True);content=archive.read(name);path.write_bytes(content);hashes[rel]=hashlib.sha256(content).hexdigest()
    for name in archive.namelist():
        if name.split('/')[-1].lower() in ['license','license.md','license.txt'] and len(name.split('/'))==2:(out/'LICENSE').write_bytes(archive.read(name))
    (out/'provenance.json').write_text(json.dumps({'commit':COMMIT,'checkpoint_url':URL,'checkpoint_sha256':SHA,'source_url':url,'source_files_sha256':hashes},indent=2))
    print('Verified NRGMark detector',len(hashes),'source files; competition package unchanged')
if __name__=='__main__':main()
