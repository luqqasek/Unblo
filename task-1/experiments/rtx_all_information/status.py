import json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent/'results';out={'data':{},'training':{},'completed':{}}
for p in r.glob('data/*/progress.json'):
 d=json.loads(p.read_text());out['data'][p.parent.name]=f"{d['completed']}/{d['total']}"
for p in r.glob('train_*.log'):
 lines=p.read_text(encoding='utf-8',errors='replace').splitlines();out['training'][p.stem]=lines[-1][:380] if lines else ''
 if any('Traceback' in s for s in lines[-20:]):out['training'][p.stem]='\n'.join(lines[-20:])
for p in r.glob('runs/*/*/fold*/completed.json'):
 d=json.loads(p.read_text());s=d['summary'];out['completed']['/'.join(p.relative_to(r/'runs').parts[:-1])]={'miou':s['all']['miou'],'tiny_fg':s['fg_le1']['fg_iou'],'tiny_fg_miss':s['fg_le1']['fg_miss']}
for p in r.glob('runs/*/completed.json'):
 d=json.loads(p.read_text());out['completed'][p.parent.name]={'delta':d['delta'],'ci':d['conditional_bootstrap95']}
out['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
print(json.dumps(out,ensure_ascii=False))
