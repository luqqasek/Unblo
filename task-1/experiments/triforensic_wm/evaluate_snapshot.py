"""Read-only evaluation of an already saved EMA; never changes training state."""
import argparse
from pathlib import Path
import torch
from .model import TriForensicWM
from .data import Engine
from .evaluate import real_evaluate,paired_evaluate,old_rectangles

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--expected-step',type=int,required=True);p.add_argument('--run',default='unified-v1');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/f'triforensic-runs/{a.family}/{a.run}';state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
    assert state['step']==a.expected_step,'Snapshot changed; do not mislabel results'
    variant='control' if a.family=='pixelseal' else 'objects';model=TriForensicWM(a.family,a.root/f'region-runs/{a.family}/{variant}-adapt/fold1/model.pt').cuda()
    model.load_state_dict(state['ema']);del state;model.eval().requires_grad_(False);engine=Engine(a.root,a.family)
    real_evaluate(torch.nn.DataParallel(model),engine,out,a.expected_step,rectangles=old_rectangles(a.root) if a.family=='trustmark' else {})
    paired_evaluate(model,engine,out,a.expected_step)
if __name__=='__main__':main()
