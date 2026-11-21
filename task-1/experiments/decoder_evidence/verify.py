"""Recompute trained/counterfactual predictions and verify blended outputs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from prepare import cv, OUT

spec=importlib.util.spec_from_file_location('evidence_run',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    torch.set_num_threads(4)
    summary=json.loads((OUT/'summary.json').read_text()); assert summary['status']=='complete'
    ds=cv.model.LocalisationDataset(cv.model.ROOT/'data','train')
    splits=json.loads((cv.OUT/'folds.json').read_text())
    for method,data in summary['methods'].items():
        cache=cv.model.extract(method,ds); names=cache['names']; features=cache['features']
        e={**torch.load(OUT/f'{method}_evidence.pt',weights_only=True),**torch.load(OUT/f'{method}_perbit.pt',weights_only=True)}
        assert e['names']==names and all(torch.isfinite(e[k]).all() for k in ['real','wrong','perbit','wrong_perbit'])
        assert len(data['runs'])==7
        unique={(r['variant'],r['fold']):r for r in data['runs']+data['cv']}
        for (variant,fold),row in unique.items():
            stem=OUT/variant/f'{method}_fold{fold}'
            split=splits[method][fold-1]
            assert row['train_names']==split['train'] and row['val_names']==split['validation']
            assert not set(row['train_names']) & set(row['val_names'])
            ids=np.array([names.index(n) for n in row['val_names']])
            with np.load(stem.with_suffix('.npz')) as z:
                assert z['names'].tolist()==row['val_names']
                np.testing.assert_array_equal(z['target'],np.stack([ds.m128[n] for n in z['names']]))
                if variant=='blend':
                    with np.load(cv.OUT/f'{method}_oof.npz') as base, np.load(OUT/'map_only'/f'{method}_fold{fold}.npz') as maps:
                        lookup={n:i for i,n in enumerate(base['names'].tolist())}
                        b=base['probability'][[lookup[n] for n in row['val_names']]]
                        for key in ('probability','counterfactual'):
                            np.testing.assert_allclose(z[key],(1-row['blend_weight'])*b+row['blend_weight']*maps[key],atol=1e-7)
                    best=max(row['inner_selection'],key=lambda x:x[0])
                    assert (row['blend_weight'],row['threshold'])==tuple(best[1:])
                else:
                    saved=torch.load(stem.with_suffix('.pt'),weights_only=True)
                    head=m.EvidenceHead(saved['channels'],variant,saved.get('evidence_channels',4)); head.load_state_dict(saved['state_dict'])
                    a,b=('perbit','wrong_perbit') if variant.startswith('bit_') else ('real','wrong')
                    if variant.endswith('_wrong'): a,b=b,a
                    for key,which in [('probability',a),('counterfactual',b)]:
                        np.testing.assert_allclose(cv.predict(head,features+[e[which]],ids),z[key],atol=1e-7)
                for key,metric in [('probability','selected'),('counterfactual','counterfactual')]:
                    np.testing.assert_allclose(cv.model.metrics(z[key],z['target'],row['threshold'])['per_image'],row[metric]['per_image'],atol=1e-12)
        if data['cv']:
            covered=[n for r in data['cv'] for n in r['val_names']]
            assert len(covered)==len(set(covered))==170
        print(method,'verified',len(unique),'runs, original and counterfactual predictions, metrics and splits',flush=True)
        if method=='stegastamp' and (OUT/'stable_summary.json').exists():
            stable=json.loads((OUT/'stable_summary.json').read_text()); assert stable['status']=='complete'
            reference=json.loads((OUT.parent.parent/'stegastamp_stable_selection/results/metrics.json').read_text())['runs']
            reference={r['fold']:r for r in reference if r['seed']==42}
            for row in stable['runs']:
                stem=OUT/'stable'/f"fold{row['fold']}"
                split=splits[method][row['fold']-1]
                assert row['train_names']==split['train'] and row['val_names']==split['validation']
                assert row['baseline']==reference[row['fold']]['miou']
                inner=[n for part in row['inner_val_names'] for n in part]
                assert len(inner)==len(set(inner)) and set(inner)==set(row['train_names'])
                ids=np.array([names.index(n) for n in row['val_names']])
                saved=torch.load(stem.with_suffix('.pt'),weights_only=True)
                h=m.EvidenceHead(saved['channels'],'bit_residual',100);h.load_state_dict(saved['state_dict'])
                with np.load(stem.with_suffix('.npz')) as z:
                    for key,which,metric in [('probability','perbit','selected'),('counterfactual','wrong_perbit','counterfactual')]:
                        p=cv.predict(h,features+[e[which]],ids)
                        np.testing.assert_allclose(p,z[key],atol=1e-7)
                        np.testing.assert_allclose(cv.model.metrics(p,z['target'],row['threshold'])['per_image'],row[metric]['per_image'],atol=1e-12)
            assert len(stable['runs'])==(3 if stable['passed'] else 1)
            print('Stable selection: verified all checkpoints, inner splits, predictions and reference scores',flush=True)


if __name__=='__main__': main()
