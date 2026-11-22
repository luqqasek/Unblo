"""One selected all-data refit if justified, followed by four inference workers."""
import json,os,subprocess,sys,time
from pathlib import Path

def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer');cfg=json.loads((root/'next-ideas-runs/selected_recipe.json').read_text());start=time.monotonic();run=root/'next-ideas-runs/stegastamp/all170'
    if cfg['needs_all170_refit'] and not (run/'completed.json').exists():
        cmd=[sys.executable,'-u','-m','experiments.next_ideas.stegastamp_adapt','--root',str(root),'--all-real','--gpus','4','--wall-seconds','600']
        if (run/'last.pt').exists():cmd+=['--resume']
        subprocess.run(cmd,check=True)
        if not (run/'completed.json').exists():print('FINAL_REFIT_PAUSED',flush=True);return
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==4;out=root/'next-ideas-production/stegastamp';out.mkdir(parents=True,exist_ok=True);workers=[];logs=[]
    for part in range(4):
        if (out/f'part{part}.json').exists():continue
        log=(out/f'worker{part}.log').open('w');logs.append(log);env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[part],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
        workers.append(subprocess.Popen([sys.executable,'-u','-m','experiments.next_ideas.infer','--root',str(root),'--part',str(part)],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in workers]
    for f in logs:f.close()
    assert not any(codes),codes;print('NEXT_IDEAS_PRODUCTION_READY',time.monotonic()-start,flush=True)
    if (root/'next-ideas-runs/extension_protocol.json').exists():
        subprocess.run([sys.executable,'-u','-m','experiments.next_ideas.extend_job'],check=True)

if __name__=='__main__':main()
