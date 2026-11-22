import os,subprocess,sys
from pathlib import Path

def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer');devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',');procs=[]
    for i,f in enumerate(['pixelseal','trustmark']):
        out=root/'region-runs'/f;out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(devices[2*i:2*i+2]))
        log=(out/'position.log').open('a')
        procs.append((f,subprocess.Popen([sys.executable,'-u','-m','experiments.region_data.position_experiment','--root',str(root),'--family',f],env=env,stdout=log,stderr=subprocess.STDOUT)))
    errors=[]
    for f,p in procs:
        if p.wait():errors.append(f)
    if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
