"""Replay metrics, threshold decisions and saved fusion checkpoints."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from probe import ROOT,OUT,incumbent,metrics,select_threshold,save
from reembed_probe import ncc
from fusion import Correction,baseline_logits,scaling


def main():
    scaling.setup('cuda:0')
    summary=json.loads((OUT/'summary.json').read_text())
    assert all(n==170 for n in summary['counts'].values())
    verified=0
    for method in ('pixelseal','trustmark'):
        source=incumbent(method)
        files=sorted((OUT/method).glob('*.json'))
        assert {p.stem for p in files}==set(source)
        for i,path in enumerate(files):
            r=json.loads(path.read_text());p,y,t,fold=source[path.stem]
            kd=r['keep_drop'];thresholds=np.array(kd['thresholds'])
            base_index=int(np.argmin(abs(thresholds-t)))
            area=np.mean(p[None]>=thresholds[:,None,None],axis=(1,2))
            for index,label in enumerate(('true','null1','null2')):
                chosen=select_threshold(np.array(kd['scores'][index]),area,base_index)
                for criterion,j in zip(('contrast','retain95'),chosen):
                    assert abs(kd[f'{criterion}_{label}_threshold']-thresholds[j])<1e-12
                    actual=scaling.cv.model.miou_image(p>=thresholds[j],y)
                    assert abs(actual-kd[f'{criterion}_{label}'])<1e-12
            if i<2:
                with np.load(path.with_suffix('.npz')) as maps:
                    measured=metrics(dict(maps),y)
                for key in measured:
                    for metric in ('auc','ap'):
                        assert abs(measured[key][metric]-r['metrics'][key][metric])<1e-12
            verified+=1
    z=np.load(ROOT/'data/train/masks_128.npz');targets=dict(zip(z['names'],z['masks']))
    for method in ('pixelseal','trustmark','stegastamp'):
        files=sorted((OUT/'reembed'/method).glob('*.json'))
        assert len(files)==170 and len({f.stem for f in files})==170
        for path in files[:2]:
            r=json.loads(path.read_text())
            with np.load(path.with_suffix('.npz')) as maps:
                measured=metrics(dict(maps),targets[path.stem])
            for key in measured:
                for metric in ('auc','ap'):
                    assert abs(measured[key][metric]-r['metrics'][key][metric])<1e-12
    # Numerical invariants for centered normalized cross correlation.
    torch.manual_seed(42);x=torch.randn(1,3,512,512,device='cuda:0')
    torch.testing.assert_close(ncc(x,x,16),torch.ones(1,128,128,device='cuda:0'),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(ncc(x,-x,32),-torch.ones(1,128,128,device='cuda:0'),atol=1e-5,rtol=1e-5)
    assert ncc(x,torch.zeros_like(x),32).abs().max()==0
    replay={}
    for method in ('pixelseal','trustmark'):
        variant={'pixelseal':'n4k_s2500','trustmark':'n4k_s5000'}[method]
        data=scaling.load_real(method,'cuda:0')
        state=torch.load(ROOT/f'experiments/synthetic_scaling/results/runs/{method}/{variant}_fold1_seed42.pt',weights_only=True)['state_dict']
        base=baseline_logits(data,state)
        maps=torch.load(OUT/'fusion'/f'{method}_maps.pt',weights_only=True)
        index={n:i for i,n in enumerate(data.names)}
        for label in ('real','wrong'):
            stem=OUT/'fusion'/f'{method}_fold1_{label}'
            ck=torch.load(stem.with_suffix('.pt'),weights_only=True)
            assert not set(ck['train_names'])&set(ck['validation_names'])
            ids=np.array([index[n] for n in ck['validation_names']])
            head=Correction().cuda();head.load_state_dict(ck['state_dict']);head.eval()
            normalized=(maps[label].cuda()-ck['mean'].cuda())/ck['std'].cuda()
            from fusion import predict
            p=predict(head,normalized,base,ids).cpu().numpy()
            expected=np.load(stem.with_suffix('.npz'))['probability']
            err=float(np.max(np.abs(p-expected)));assert err<=1e-7
            replay[method+'_'+label]=err
        del data,maps;torch.cuda.empty_cache()
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}
    save(OUT/'verification.json',dict(status='passed',threshold_images_replayed=verified,
         map_metric_replay='2 images per method and family; exact AUC/AP',
         ncc_invariants='identity +1, negative -1, zero 0',fusion_max_probability_errors=replay,
         counts=summary['counts'],code_sha256=hashes))
    print('VERIFIED: 340 threshold decisions, map metric replay, NCC invariants, 4 fusion checkpoints.')


if __name__=='__main__':main()
