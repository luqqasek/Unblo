"""Foreground coordinator: preparation must finish before occupying RTX3070."""
import os,sys,time,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def main():
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONUNBUFFERED='1')
    results=HERE/'results';results.mkdir(exist_ok=True)
    for f in ('pixelseal','trustmark','stegastamp'):
     while not (results/'data'/f/'complete.json').exists():
      p=results/'data'/f/'progress.json'
      print('WAITING_FOR_DATA',f,p.read_text().replace('\n',' ') if p.exists() else 'not started',flush=True)
      time.sleep(30)
     with (results/f'train_{f}.log').open('a',encoding='utf-8') as log:
      cmd=[sys.executable,'-X','utf8','-u','-m','experiments.rtx_all_information.train','--family',f]
      print('START_FAMILY',f,flush=True)
      p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
      if p.returncode:print('FAILED',f,log.name,flush=True);sys.exit(p.returncode)
     print('FINISHED_FAMILY',f,flush=True)
    print('ALL_FAMILIES_COMPLETE',flush=True)

if __name__ == "__main__":
    main()
