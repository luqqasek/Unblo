"""Fetch pinned official source/weights; no model execution on login nodes."""
import argparse,subprocess,json
from pathlib import Path
import gdown
from .model import PIN,WEIGHT_HASH
from experiments.native_localizer.common import sha256,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();source=a.root/'research-source/iml-vit';weights=a.root/'research-weights/iml-vit/iml-vit_checkpoint.pth'
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['git','clone','https://github.com/SunnyHaze/IML-ViT.git',str(source)],check=True);subprocess.run(['git','-C',str(source),'checkout',PIN],check=True)
    if (source/'.git').exists():assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==PIN
    if not weights.exists():
        weights.parent.mkdir(parents=True,exist_ok=True);gdown.download(id='1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7',output=str(weights),quiet=False)
    assert sha256(weights)==WEIGHT_HASH
    files=[source/'iml_vit_model.py',*sorted((source/'modules').glob('*.py')),source/'LICENSE']
    atomic_json(a.root/'research-weights/iml-vit/provenance.json',{'repository':'https://github.com/SunnyHaze/IML-ViT','commit':PIN,'weights_sha256':WEIGHT_HASH,'license_confirmation':'https://github.com/SunnyHaze/IML-ViT/issues/60','files':{str(f.relative_to(source)):sha256(f) for f in files}})
if __name__=='__main__':main()
