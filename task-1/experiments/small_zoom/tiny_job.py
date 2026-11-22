import os,subprocess,sys
from pathlib import Path
def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer')
    if '--adapt' not in sys.argv:
        from .tiny_data import prepare
        prepare(root)
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');procs=[]
    for i,v in enumerate(['control','tiny','tiny_aux']):
        out=root/'tiny-runs'/v;out.mkdir(parents=True,exist_ok=True);log=(out/('adapt.log' if '--adapt' in sys.argv else 'run.log')).open('w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i]);cmd=[sys.executable,'-u','-m','experiments.small_zoom.tiny_train','--root',str(root),'--variant',v]
        if '--adapt' in sys.argv:cmd+=sys.argv[1:]
        procs.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in procs]
    if any(codes):raise RuntimeError(codes)
if __name__=='__main__':main()
