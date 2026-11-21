"""Replicate frozen StegaStamp pilot for another exact watermark family."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from .common import PROJECT, atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--family',choices=['pixelseal','trustmark'],required=True)
    p.add_argument('--stage',choices=['generate','train','all'],default='all')
    a=p.parse_args()
    start=time.monotonic()
    out=a.root/f'runs/{a.family}/pilot-30min'
    def run(module,*args):
        print('STAGE',module,*args,flush=True)
        subprocess.run([sys.executable,'-m',f'experiments.native_localizer.{module}',*map(str,args)],check=True)
    if a.stage in ('generate','all'):
        for split,count in [('train',2500),('dev',128)]:
            run('generator','--sources',a.root/f'sources/coco2017/{split}.jsonl','--partition',split,
                '--method',a.family,'--count',count,'--models','checkpoints',
                '--out',a.root/f'encoded/{a.family}/pilot-{split}')
    if a.stage in ('train','all'):
        if (out/'last.pt').exists():
            raise FileExistsError('Preserving trained model')
        if time.monotonic()-start>16*60:
            print('GENERATION_COMPLETE_TRAIN_IN_NEXT_ALLOCATION',flush=True)
            return
        cfg=json.loads((PROJECT/'pilot_config_frozen.json').read_text())
        cfg['family']=a.family
        config=a.root/f'pilot-{a.family}-config.json'
        atomic_json(config,cfg)
        run('train','--config',config,'--family',a.family,
            '--train',a.root/f'encoded/{a.family}/pilot-train/encoded.jsonl',
            '--dev',a.root/f'encoded/{a.family}/pilot-dev/encoded.jsonl',
            '--out',out,'--backbone','checkpoints/convnext_base.fb_in22k/model.safetensors',
            '--gpus',4,'--dev-limit',64,'--real-fold',1,'--real-data','data')
        run('real_eval','--checkpoint',out/'last.pt','--data','data','--run',out,'--folds',2,3)
    atomic_json(a.root/f'{a.family}-pilot-{a.stage}-completed.json',{'seconds':time.monotonic()-start,'family':a.family})

if __name__=='__main__':
    main()
