"""Fetch pinned official sources and verified checkpoints, without replacing existing files."""
import sys
import subprocess
import requests
from common import *

def download(url,path):
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    part=path.with_suffix(path.suffix+'.partial')
    with requests.get(url,stream=True,timeout=180) as r:
        r.raise_for_status()
        with part.open('wb') as f:
            for chunk in r.iter_content(2**20):f.write(chunk)
    part.rename(path)

def main():
    # SAM2 SHA is captured from the actual integration checkout by finalize/verify.
    pins=json.loads((HERE/'source_pins.json').read_text())
    for name,info in pins.items():
        path=OUT/'vendor'/name
        if not path.exists():
            subprocess.run(['git','clone',info['url'],str(path)],check=True)
            subprocess.run(['git','-C',str(path),'checkout',info['commit']],check=True)
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        assert actual==info['commit'],(name,actual)
    download('https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip',OUT/'TruFor_weights.zip')
    with (OUT/'TruFor_weights.zip').open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()=='7bee48f3476c75616c3c5721ab256ff8'
    base='https://huggingface.co/Gertlek/DetectiveSAM/resolve/main/'
    for file,hash_ in [('checkpoints/detective_sam_sota.pth','0c7c33c4c4ddb82459bc2b6bbfdfd1003561e2714d02163b3daffb25ede4e1cf'),
                       ('sam2configs/sam2.1_hiera_base_plus.pt','a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5')]:
        path=OUT/'vendor/DetectiveSAM'/file;download(base+file,path);assert sha(path)==hash_
    mesorch=OUT/'weights/mesorch-98.pth';mesorch.parent.mkdir(exist_ok=True)
    if not mesorch.exists():
        sys.path.insert(0,str(OUT/'deps'));import gdown
        gdown.download(id='1PJxKteinMyaAYokKy0JhuzBnBc6bGsau',output=str(mesorch),quiet=False)
    assert mesorch.stat().st_size==1023886070
    expected=json.loads((HERE/'checkpoint_hashes.json').read_text())
    assert sha(mesorch)==expected['mesorch-98.pth']
    print('Official source pins and checkpoint hashes verified.')

if __name__=='__main__':main()
