"""Cache metrics for completed fits; no checkpoint selection and no training."""
from scaling_common import *

def main():
    report=module('scaling_ready_report',HERE/'report.py')
    meta={r['image_name']:r for r in json.loads((ROOT/'data/train/metadata.json').read_text())}
    count=0
    for p in sorted((OUT/'runs').glob('*/*_fold*_seed*.npz')):
        if p.with_suffix('.json').exists():report.measured_run(p,meta);count+=1
    print('Metric cache for',count,'complete outer fits',flush=True)

if __name__=='__main__':main()
