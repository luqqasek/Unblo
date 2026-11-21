"""Recompute masks, bit diagnostics and verify the unchanged control."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('message_run',HERE/'run.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    torch.set_num_threads(4)
    s=json.loads((HERE/'results/summary.json').read_text())
    assert s['status']=='complete' and set(s['variants'])==set(m.VARIANTS)
    ds=m.cv.model.LocalisationDataset(m.cv.model.ROOT/'data','train')
    splits=json.loads((m.cv.OUT/'folds.json').read_text())
    baseline=json.loads((m.cv.OUT/'metrics.json').read_text())
    for method in m.cv.METHODS:
        cache=m.cv.model.extract(method,ds); names=cache['names']; features=cache['features']
        lookup={x['image_name']:x for x in ds.meta}
        targets=np.stack([ds.m128[n] for n in names])
        coverage=F.avg_pool2d(torch.from_numpy(targets).float()[:,None],4)[:,0]
        messages=torch.tensor([[int(c) for c in lookup[n]['message']] for n in names],dtype=torch.float32)
        for variant,data in s['variants'].items():
            rows=data['cv'] or data['screen']
            for row in [r for r in rows if r['method']==method]:
                stem=HERE/'results'/variant/f"{method}_fold{row['fold']}"
                split=splits[method][row['fold']-1]
                assert row['train_names']==split['train'] and row['val_names']==split['validation']
                assert not set(row['train_names']) & set(row['val_names'])
                ids=np.array([names.index(n) for n in row['val_names']])
                fit=np.array([names.index(n) for n in row['train_names']])
                checkpoint=torch.load(stem.with_suffix('.pt'),weights_only=True)
                head=m.JointHead(checkpoint['channels'],checkpoint['bits']); head.load_state_dict(checkpoint['state_dict'])
                prob=m.cv.predict(head,features,ids)
                with np.load(stem.with_suffix('.npz')) as z:
                    assert z['names'].tolist()==row['val_names']
                    np.testing.assert_array_equal(z['target'],targets[ids])
                    np.testing.assert_allclose(prob,z['probability'],atol=1e-7)
                np.testing.assert_allclose(m.cv.model.metrics(prob,targets[ids],row['threshold'])['per_image'],row['selected']['per_image'],atol=1e-12)
                d=m.diagnostics(head,features,ids,messages,coverage,fit)
                for key in d: np.testing.assert_allclose(d[key],row['diagnostics'][key],atol=1e-7)
                if variant=='control':
                    np.testing.assert_allclose(row['selected']['per_image'],baseline['methods'][method]['folds'][0]['selected']['per_image'],atol=1e-7)
            print(method,variant,'verified',flush=True)
    for variant,data in s['variants'].items():
        delta=[r['selected']['miou']-baseline['methods'][r['method']]['folds'][0]['selected']['miou'] for r in data['screen']]
        assert data['passed']==bool(variant!='control' and np.mean(delta)>=.005 and sum(d>0 for d in delta)>=2)
        if data['cv']:
            for method in m.cv.METHODS:
                names=[n for r in data['cv'] if r['method']==method for n in r['val_names']]
                assert len(names)==len(set(names))==170
    print('Verified checkpoints, predictions, message diagnostics, split coverage and gate.')


if __name__=='__main__': main()
