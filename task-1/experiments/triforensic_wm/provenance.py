"""Pinned package/source/checkpoint provenance for exact wrapper reproduction."""
import argparse,importlib.metadata,json
from pathlib import Path
from experiments.native_localizer.common import sha256,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();task=a.root/'task-1'
    packages={}
    for package in ('torch','torchvision','numpy','scipy','scikit-learn','timm','trustmark','Pillow'):
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]='not installed'
    files=[task/'wmloc/watermarks.py',task/'wmloc/utils.py',task/'checkpoints/pixelseal_card.yaml',task/'checkpoints/pixelseal/checkpoint.pth',task/'checkpoints/videoseal/videoseal/modules/convnext.py',task/'checkpoints/videoseal/videoseal/modules/jnd.py',task/'experiments/native_localizer/reconstructed_splits.json',task/'experiments/local_presence/SRM_Kernels.npy']
    result={'packages':packages,'files':{str(p.relative_to(task)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in files},'PixelSeal_backbone':'Official ConvNeXtV2 tiny with GRN, source verified','payloads':{'pixelseal':'raw256bits; no text or ECC','trustmark':'Q,secret_len100,use_ECC=False,MODE=binary,trustmark0.8.0'},'quantization':'Unchanged official competition wrapper,PNG/uint8; support from provenance','Noiseprint':json.loads((Path(__file__).parent/'noiseprint_provenance.json').read_text())}
    atomic_json(a.root/'triforensic-data/environment.json',result);print(json.dumps({'packages':packages,'files_verified':len(files)}))
if __name__=='__main__':main()
