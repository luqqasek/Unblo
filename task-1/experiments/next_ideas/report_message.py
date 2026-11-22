import json
from pathlib import Path
from experiments.native_localizer.common import atomic_json

def main():
    root=Path(__file__).parent;r=root/'results/message_extended';d=json.loads((r/'evaluation.json').read_text());lines=['# Bounded PS/TM continuation to450: measured results','','Four workers on one4-A100 node, restored parent150 model/optimizer/RNG,300 additional updates, fixed original fold1, unchanged architecture/loss/LRs and effective batch16 (12 exact synthetic +4 real). Fixed128 replay versus existing online replay. No input augmentation or new embedding calls.','', 'The TM primary comparison uses the unchanged integer90/100 verifier and caps accepted rectangles to15% image area at both endpoints. Historical uncapped maps are also retained for reproduction. It does not include the later empty-mask scan, so these TM numbers must not be called the full#2675 recipe. No test inference was run for these arms.','', '| Arm | mIoU150 | mIoU450 | Delta | Paired95% CI | Candidate |','|---|---:|---:|---:|---|---|'];measure=[]
    for name,v in d['results'].items():
        a,b=v['before'],v['after'];lines.append(f"| {name} | {a['all']['miou']:.6f} | {b['all']['miou']:.6f} | {v['delta']:+.6f} | {v['ci95']} | {v['candidate']} |")
        measure.append({'family':'pixelseal' if name.startswith('ps') else 'trustmark','name':name+' / continuation450','protocol':'Same57-image fold1; existing150 parent; exploratory repeated development fold; fixed capped TM verifier','miou':b['all']['miou'],'note':f"Delta{v['delta']:+.6f}; candidate{v['candidate']}; no submission."})
    lines+=['','| Arm | Bucket/metric | n | Before150 | After450 | Misses before / after |','|---|---|---:|---:|---:|---|']
    for name,v in d['results'].items():
        for g,k,miss in [('fg_le1','fg_iou','fg_miss'),('fg_1to5','fg_iou','fg_miss'),('fg_5to20','fg_iou','fg_miss'),('fg_gt20','fg_iou','fg_miss'),('clean_le1','clean_iou','clean_miss'),('clean_1to5','clean_iou','clean_miss'),('wm_over50','clean_iou','clean_miss')]:
            a,b=v['before'][g],v['after'][g];lines.append(f"| {name} | {g}/{k} | {b['n']} | {a[k]:.6f} | {b[k]:.6f} | {a[miss]}/{b[miss]} |")
    passed=[n for n,v in d['results'].items() if v['candidate']]
    lines+=['',f"Arms passing the predeclared +0.005 gain, positive paired lower confidence bound and minority guards: {passed or 'none'}. The confidence intervals do not correct for repeated fold reuse or four-arm selection. The best accepted submission remains#3054/public0.6677 unless a later actual server receipt establishes otherwise.", '',f"Orchestrator wall seconds: {d['seconds']:.2f}. Additional optimizer updates1200, total19200 image views (14400 synthetic replay +4800 real draws). Parent probability reproduction is checked within1e-4 before training; logs include measured throughput and allocated memory. No corpus expansion was performed."]
    (root/'MESSAGE_EXTENSION_REPORT.md').write_text('\n'.join(lines)+'\n');atomic_json(r/'measurements.json',measure)
if __name__=='__main__':main()
