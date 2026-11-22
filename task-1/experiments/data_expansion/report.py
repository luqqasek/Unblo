"""Append measured screen results without conflating preparation with model gains."""
import json
from pathlib import Path


def main():
    here=Path(__file__).parent;out=here/'results';p=out/'runs/evaluation.json'
    if not p.exists():return
    r=json.loads(p.read_text());path=here/'REPORT.md';text=path.read_text().split('\n## Measured screen')[0]
    text=text.replace('No PIPE training score is available yet.','Measured training results are recorded below.')
    lines=['','## Measured screen','','| Arm | mIoU | Delta vs initial |','|---|---:|---:|']
    baseline=r['baseline']['all']['miou'];measures=[]
    for key in ('baseline','old_replay','pipe'):
        value=r[key]['all']['miou'];lines.append(f'| {key} | {value:.6f} | {value-baseline:+.6f} |')
        measures.append({'family':'pixelseal','variant':key,'protocol':'Matched200-update screen on reused57-image fold; current EMA1561 parent','miou':value,'note':f'Delta vs initial{value-baseline:+.6f}; PIPE selected={r["selected"]}'})
    lines+=['',f"PIPE vs initial: `{json.dumps(r['pipe_vs_initial'])}`.",f"PIPE vs matched old replay: `{json.dumps(r['pipe_vs_old'])}`.",'',f"Passed predeclared gate: **{r['selected']}**. These are reused-fold developmental measurements, not full OOF or public scores.",'','| Group | n | Metric | Initial | Old replay | PIPE | Initial/old/PIPE misses |','|---|---:|---|---:|---:|---:|---|']
    for group,metric,miss in [('fg_le1','fg_iou','fg_miss'),('fg_1to5','fg_iou','fg_miss'),('fg_5to20','fg_iou','fg_miss'),('fg_gt20','fg_iou','fg_miss'),('clean_le1','clean_iou','clean_miss'),('clean_1to5','clean_iou','clean_miss'),('wm_over50','clean_iou','clean_miss')]:
        a,b,c=[r[k][group] for k in ('baseline','old_replay','pipe')]
        lines.append(f"| {group} | {a['n']} | {metric} | {a[metric]:.6f} | {b[metric]:.6f} | {c[metric]:.6f} | {a[miss]}/{b[miss]}/{c[miss]} |")
    done=out/'runs/completed.json'
    if done.exists():
        d=json.loads(done.read_text());lines+=['',f"Shared allocation{d['job_id']}, two GPUs; encode + train + evaluate wall time{d['seconds']/60:.2f} minutes.200optimizer updates and3200training image views per arm.576exact embedding calls shared across arms (512train,64dev);64locked sources remain unencoded and unevaluated."]
    if not r['selected']:lines+=['','No test inference or leaderboard submission uses this failed data screen. This result does not establish that PIPE is useless: it rejects this bounded mixture/adaptation recipe. It also does not justify scaling the corpus without a new diagnosed hypothesis.']
    path.write_text(text+'\n'.join(lines)+'\n');(out/'measurements.json').write_text(json.dumps(measures,indent=2)+'\n')


if __name__=='__main__':main()
