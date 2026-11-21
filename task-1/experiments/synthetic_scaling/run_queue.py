"""Two local GPU queues; wait for complete data, then run fixed predeclared experiments."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=[0,1],required=True);args=ap.parse_args()
    methods=['pixelseal','trustmark'] if args.gpu==0 else ['stegastamp']
    for method in methods:
        requirements=[OUT/'data'/method/'complete.json']
        if args.gpu==1:requirements.append(OUT/'data/trustmark/complete.json')
        while not all(p.exists() for p in requirements):
            print(f'QUEUE gpu={args.gpu} waiting for generation: {method}',flush=True);time.sleep(30)
        logfile=OUT/f'training_{method}.log'
        print(f'START TRAINING {method} gpu={args.gpu}; log={logfile}',flush=True)
        with logfile.open('a',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--method',method,'--device',f'cuda:{args.gpu}'],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(f'FINISHED {method}',flush=True)

if __name__=='__main__':main()
