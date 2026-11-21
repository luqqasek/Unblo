import os,subprocess,sys
from pathlib import Path

def main():
 devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',');procs=[]
 for i,family in enumerate(('pixelseal','trustmark')):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(devices[2*i:2*i+2]));out=Path('../message-runs')/family;out.mkdir(parents=True,exist_ok=True)
  log=(out/'matched-v1-adapt.log').open('a')
  args=[sys.executable,'-u','-m','experiments.message_localizer.adapt','--family',family,'--root','/p/project1/training2666/pszenny1/native-localizer','--steps','150','--gpus','2']
  procs.append((family,subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)))
 errors=[]
 for f,p in procs:
  code=p.wait();print(f,'exit',code,flush=True)
  if code:errors.append(f)
 if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
