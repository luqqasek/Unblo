"""Refit the selected endpoint in one loop, then export and infer on four GPUs.

An allocation can end between stages or pause the same optimizer trajectory.
Training never starts from an intermediate TriForensic screening checkpoint.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['pixelseal','trustmark'],required=True)
    p.add_argument('--root',type=Path,default=Path('/p/project1/training2666/pszenny1/native-localizer'))
    a=p.parse_args();start=time.monotonic();screen=a.root/f'triforensic-runs/{a.family}/unified-v1'
    run=a.root/f'triforensic-runs/{a.family}/all170-v1';out=a.root/f'triforensic-production/{a.family}'
    done=json.loads((screen/'completed.json').read_text())
    assert done['gate']['passed'] and done['stop'] in ('steps_complete','user_authorized_early_stop')
    if not (run/'completed.json').exists():
        cmd=[sys.executable,'-u','-m','experiments.triforensic_wm.train','--root',str(a.root),'--family',a.family,'--run','all170-v1','--all-real','--screening',str(screen),'--wall-seconds','1600']
        if (run/'resume.pt').exists():cmd+=['--resume']
        subprocess.run(cmd,check=True)
        if not (run/'completed.json').exists():print('REFIT_PAUSED_SAME_LOOP',flush=True);return
    if not (run/'frozen_inference.json').exists():subprocess.run([sys.executable,'-m','experiments.triforensic_wm.export','--run',str(run)],check=True)
    if time.monotonic()-start>1350:
        print('REFIT_COMPLETE_INFERENCE_NEXT_ALLOCATION',flush=True);return
    out.mkdir(parents=True,exist_ok=True);devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==4
    workers=[];logs=[]
    for part in range(4):
        if (out/f'part{part}.json').exists() and (out/f'part{part}.pt').exists():continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[part],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
        log=(out/f'worker{part}.log').open('w');logs.append(log)
        workers.append(subprocess.Popen([sys.executable,'-u','-m','experiments.triforensic_wm.infer','--root',str(a.root),'--recipe',str(run/'frozen_inference.json'),'--out',str(out),'--part',str(part),'--test'],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[worker.wait() for worker in workers]
    for log in logs:log.close()
    assert not any(codes),codes
    print('PRODUCTION_INFERENCE_COMPLETE',a.family,flush=True)


if __name__=='__main__':main()
