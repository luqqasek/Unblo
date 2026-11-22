import os,subprocess,sys
from pathlib import Path
def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer');out=root/'local-presence-runs';dev=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(dev)==4
    jobs=[('ps-spatial-dice','spatial',['--loss','dice']),('ps-spatial-iou','spatial',['--loss','iou']),('tm-scan-0','scan_empty',['--part','0']),('tm-scan-1','scan_empty',['--part','1'])];procs=[]
    for i,(name,module,args) in enumerate(jobs):
        log=(out/f'{name}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=dev[i],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6');procs.append((name,subprocess.Popen([sys.executable,'-u','-m','experiments.local_presence.'+module,'--root',str(root),*args],env=env,stdout=log,stderr=subprocess.STDOUT)))
    errors=[]
    for name,p in procs:
        rc=p.wait();print('WORKER_DONE',name,rc,flush=True)
        if rc:errors.append((name,rc))
    if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
