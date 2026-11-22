"""Continue the SAME loop, or start next family after a completed PS screen."""
import json,os,subprocess,sys,time
from pathlib import Path

def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer');started=time.monotonic();ps=root/'triforensic-runs/pixelseal/unified-v1'
    if not (ps/'completed.json').exists():
        assert (ps/'resume.pt').exists(),'No PS state to resume'
        subprocess.run([sys.executable,'-u','-m','experiments.triforensic_wm.train','--root',str(root),'--family','pixelseal','--resume','--carrier-no-checkpointing'],check=True)
        return
    complete=json.loads((ps/'completed.json').read_text());print('PS_SCREEN_STATE',complete['stop'],complete['step'],flush=True)
    tm=root/'triforensic-runs/trustmark/unified-v1'
    if (tm/'completed.json').exists():print('Both screens complete; no new training launched',flush=True);return
    ready=all((root/f'triforensic-data/trustmark/{partition}/shard{i}/completed.json').exists() for partition in ('train','dev') for i in range(4))
    if not ready:subprocess.run([sys.executable,'-u','-m','experiments.triforensic_wm.prepare_job','--family','trustmark'],check=True)
    budget=1650-int(time.monotonic()-started)
    if budget<300:print('TM_DATA_READY; insufficient safe time for optimizer loop',flush=True);return
    command=[sys.executable,'-u','-m','experiments.triforensic_wm.train','--root',str(root),'--family','trustmark','--wall-seconds',str(budget),'--carrier-no-checkpointing']
    if (tm/'resume.pt').exists():command+=['--resume']
    subprocess.run(command,check=True)
if __name__=='__main__':main()
