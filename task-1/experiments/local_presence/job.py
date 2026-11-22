"""Pack four independent GPU workers into one and only one Slurm node."""
import os,subprocess,sys,time
from pathlib import Path
def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer');out=root/'local-presence-runs';out.mkdir(exist_ok=True)
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==4,devices
    jobs=[('pixelseal-presence',['train','--family','pixelseal']),('trustmark-presence',['train','--family','trustmark']),('pixelseal-holes',['train','--family','pixelseal','--holes']),('tm-tiled',['tiled'])]
    procs=[]
    for i,(name,args) in enumerate(jobs):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i],OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6');log=(out/f'{name}.log').open('w')
        cmd=[sys.executable,'-u','-m',f'experiments.local_presence.{args[0]}','--root',str(root),*args[1:]]
        procs.append((name,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)))
    errors=[]
    while procs:
        for name,p in procs[:]:
            if p.poll() is not None:
                print('WORKER_FINISHED',name,p.returncode,flush=True)
                if p.returncode:errors.append((name,p.returncode))
                procs.remove((name,p))
        if procs:time.sleep(5)
    if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
