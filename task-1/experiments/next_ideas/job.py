"""One node: three independent outer-fold workers plus frozen oracle diagnostics."""
import os,subprocess,sys
from pathlib import Path

def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer');logs=root/'next-ideas-logs';logs.mkdir(exist_ok=True);devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==4;workers=[];handles=[]
    for gpu in range(4):
        module='stegastamp_adapt' if gpu<3 else 'oracle_ladder';args=['--fold',str(gpu+1)] if gpu<3 else []
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6');f=(logs/f'worker{gpu}.log').open('w');handles.append(f)
        workers.append(subprocess.Popen([sys.executable,'-u','-m','experiments.next_ideas.'+module,'--root',str(root),*args],env=env,stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in workers]
    for f in handles:f.close()
    print('NEXT_IDEAS_WORKERS',codes,flush=True);assert not any(codes)

if __name__=='__main__':main()
