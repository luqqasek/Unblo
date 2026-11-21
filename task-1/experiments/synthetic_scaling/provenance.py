"""Snapshot pretrained sources and retained best artifacts, without credentials."""
import importlib.metadata
import trustmark
from scaling_common import *

def main():
    paths=[ROOT/'wmloc/watermarks.py',ROOT/'wmloc/utils.py',ROOT/'checkpoints/stegastamp/stega_stamp_logits.onnx',
           ROOT/'checkpoints/pixelseal/checkpoint.pth',ROOT/'checkpoints/pixelseal_card.yaml',
           OLD/'results/full_train/submission.pt',OLD/'results/full_train/stegastamp_head.pt',
           HERE.parent/'frozen_extractor/results/cv3/folds.json',HERE.parent/'stegastamp_stn_alignment/results/splits.json']
    package=Path(trustmark.__file__).parent
    paths += [package/'models'/n for n in ('encoder_Q.ckpt','decoder_Q.ckpt','trustmark_Q.yaml')]
    paths += [real_source(m) for m in METHODS]
    snapshot=dict(files={str(p.resolve()):sha(p) for p in paths},
                  packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','trustmark','onnxruntime','timm')},
                  note='Snapshot before training; running generation uses the same retained pretrained sources and verified feature hooks.')
    path=OUT/'source_snapshot.json'
    if path.exists():assert json.loads(path.read_text())==snapshot
    else:save(path,snapshot)
    print('SOURCE SNAPSHOT',len(paths),'files',flush=True)

if __name__=='__main__':main()
