"""Full-data refits followed by test inference, one pair of GPUs per family."""
import os, subprocess, sys, argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inference-only',action='store_true');args=parser.parse_args()
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    procs=[]
    for i,family in enumerate(('pixelseal','trustmark')):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(devices[2*i:2*i+2]))
        log=(Path('../message-runs')/family/'full-refit.log').open('a')
        if args.inference_only:
            procs.append((family,env,log,None));continue
        procs.append((family,env,log,subprocess.Popen([sys.executable,'-u','-m','experiments.message_localizer.adapt','--family',family,'--root','/p/project1/training2666/pszenny1/native-localizer','--steps','150','--gpus','2','--full-refit'],env=env,stdout=log,stderr=subprocess.STDOUT)))
    for family,env,log,proc in procs:
        if proc is not None and proc.wait():raise RuntimeError(f'{family} refit failed')
    procs=[(f,subprocess.Popen([sys.executable,'-u','-m','experiments.message_localizer.infer','--family',f,'--root','/p/project1/training2666/pszenny1/native-localizer'],env=e,stdout=l,stderr=subprocess.STDOUT)) for f,e,l,_ in procs]
    for f,p in procs:
        if p.wait():raise RuntimeError(f'{f} inference failed')
if __name__=='__main__': main()
