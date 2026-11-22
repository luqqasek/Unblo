import os,subprocess,sys
from pathlib import Path
def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer');out=root/'local-presence-production';out.mkdir(exist_ok=True);dev=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(dev)==4;procs=[]
    for i in range(4):
        log=(out/f'worker-{i}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=dev[i],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6');procs.append(subprocess.Popen([sys.executable,'-u','-m','experiments.local_presence.production','--root',str(root),'--part',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in procs];print('PRODUCTION_WORKERS',codes,flush=True);assert codes==[0]*4
if __name__=='__main__':main()
