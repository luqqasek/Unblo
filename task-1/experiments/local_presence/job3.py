"""Four independent fixed-budget screens, one physical node."""
import os,subprocess,sys
from pathlib import Path

def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer');out=root/'local-presence-runs';dev=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(dev)==4
    jobs=[('ps-adapt-control','pixelseal',False),('ps-adapt-real','pixelseal',True),('tm-adapt-control','trustmark',False),('tm-adapt-real','trustmark',True)];procs=[]
    for i,(name,family,real) in enumerate(jobs):
        log=(out/f'{name}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=dev[i],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
        args=[sys.executable,'-u','-m','experiments.local_presence.adapt_branch','--root',str(root),'--family',family]+(['--real'] if real else [])
        procs.append((name,subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)))
    errors=[]
    for name,p in procs:
        rc=p.wait();print('WORKER_DONE',name,rc,flush=True)
        if rc:errors.append((name,rc))
    if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
