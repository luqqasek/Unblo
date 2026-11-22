import argparse,os,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--family',required=True);a=p.parse_args();root=Path('/p/project1/training2666/pszenny1/native-localizer');out=root/'triforensic-logs';out.mkdir(exist_ok=True);dev=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(dev)==4;workers=[]
    for i in range(4):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=dev[i],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6');log=(out/f'prepare-{a.family}-{i}.log').open('w');workers.append(subprocess.Popen([sys.executable,'-u','-m','experiments.triforensic_wm.prepare','--root',str(root),'--family',a.family,'--part',str(i)],stdout=log,stderr=subprocess.STDOUT,env=env))
    codes=[p.wait() for p in workers];print('PREPARE_DONE',a.family,codes,flush=True);assert codes==[0]*4
if __name__=='__main__':main()
