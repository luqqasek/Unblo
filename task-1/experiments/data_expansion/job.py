"""Two GPU workers only: encode disjoint shards, then matched200-update arms."""
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.native_localizer.common import atomic_json


def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer')
    out=root/'data-expansion/runs';out.mkdir(parents=True,exist_ok=True)
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==2
    start=time.monotonic()
    for module,arguments in [('encode',[['--part','0'],['--part','1']]),('train',[['--arm','old'],['--arm','pipe']])]:
        processes=[];handles=[]
        for i,args in enumerate(arguments):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
            log=(out/f'{module}{i}.log').open('a');handles.append(log)
            processes.append(subprocess.Popen([sys.executable,'-u','-m',f'experiments.data_expansion.{module}','--root',str(root),*args],env=env,stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in processes]
        for f in handles:f.close()
        assert not any(codes),(module,codes)
    subprocess.run([sys.executable,'-m','experiments.data_expansion.evaluate','--root',str(root)],check=True)
    atomic_json(out/'completed.json',{'seconds':time.monotonic()-start,'job_id':os.environ.get('SLURM_JOB_ID'),'gpus':2})


if __name__=='__main__':main()
