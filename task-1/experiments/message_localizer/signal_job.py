import os,subprocess,sys
from pathlib import Path

def main():
 devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(','); procs=[]
 for i,family in enumerate(('pixelseal','trustmark')):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i]);out=Path('../message-signal')/family;out.mkdir(parents=True,exist_ok=True)
  log=(out/'run.log').open('a')
  procs.append((family,subprocess.Popen([sys.executable,'-u','-m','experiments.message_localizer.signal_check','--family',family,'--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)))
 for family,p in procs:
  code=p.wait();print(family,'exit',code,flush=True)
  if code:raise RuntimeError(f'{family} failed')
if __name__=='__main__':main()
