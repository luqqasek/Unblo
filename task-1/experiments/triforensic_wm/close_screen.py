"""Freeze a saved endpoint after the user's explicit early-stop instruction."""
import argparse,json
from pathlib import Path
import torch
from experiments.native_localizer.common import atomic_json,sha256
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reason',required=True);a=p.parse_args()
    assert not (a.run/'completed.json').exists();paused=json.loads((a.run/'paused.json').read_text());cfg=json.loads((a.run/'config.json').read_text())
    state=torch.load(a.run/'resume.pt',map_location='meta',weights_only=False);assert state['step']==paused['step'];del state
    rows=[json.loads(s) for s in (a.run/'train.jsonl').read_text().splitlines()];discarded=[r for r in rows if r['step']>paused['step']]
    if discarded:
        (a.run/'discarded-performance-attempt.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in discarded))
        (a.run/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=paused['step']))
    paused.update(stop='user_authorized_early_stop',reason=a.reason,planned_schedule_updates=3500,retained_checkpoint_sha256=sha256(a.run/'resume.pt'),discarded_logged_updates=max([r['step']-paused['step'] for r in discarded],default=0),discarded_updates_note='At least this many; cancellation may include up to24 later unlogged updates. Excluded from retained optimizer trajectory, recorded as compute overhead.')
    atomic_json(a.run/'completed.json',paused);atomic_json(a.run/'frozen_selection.json',{'step':paused['step'],'schedule_total_steps':cfg['steps'],'threshold':.5,'family':cfg['family'],'reason':a.reason,'policy':'User explicitly requested early stopping where targeted metrics do not improve; one reused development fold, adaptive selection disclosed.'});print(json.dumps({'step':paused['step'],'miou':paused['final']['all']['miou'],'stop':paused['stop']}))
if __name__=='__main__':main()
