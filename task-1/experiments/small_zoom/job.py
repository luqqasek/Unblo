import os,sys,subprocess
from pathlib import Path
def main():
 root=Path('/p/project1/training2666/pszenny1/native-localizer');devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');procs=[]
 for i,f in enumerate(['pixelseal','trustmark']):
  out=root/'small-zoom-runs'/f;out.mkdir(parents=True,exist_ok=True)
  log=(out/'proposals.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i])
  procs.append(subprocess.Popen([sys.executable,'-u','-m','experiments.small_zoom.proposals','--root',str(root),'--family',f]+sys.argv[1:],env=env,stdout=log,stderr=subprocess.STDOUT))
 codes=[p.wait() for p in procs]
 if any(codes):raise RuntimeError(codes)
if __name__=='__main__':main()
