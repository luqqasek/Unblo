"""Verify saved loss experiments and conditional CV coverage."""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('loss_run',HERE/'run.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    s=json.loads((HERE/'results/summary.json').read_text())
    assert s['status']=='complete' and set(s['variants'])==set(m.VARIANTS)
    splits=json.loads((m.cv.OUT/'folds.json').read_text())
    ds=m.cv.model.LocalisationDataset(m.cv.model.ROOT/'data','train')
    for variant,v in s['variants'].items():
        delta=[r['selected']['miou']-s['baseline']['methods'][r['method']]['folds'][0]['selected']['miou'] for r in v['screen']]
        assert v['passed']==bool(np.mean(delta)>=.005 and sum(x>0 for x in delta)>=2)
        rows=v['cv'] if v['passed'] else v['screen']
        assert len(rows)==(9 if v['passed'] else 3)
        for r in rows:
            split=splits[r['method']][r['fold']-1]
            assert r['train_names']==split['train'] and r['val_names']==split['validation']
            assert not set(r['train_names']) & set(r['val_names'])
            with np.load(HERE/'results'/variant/f"{r['method']}_fold{r['fold']}.npz") as z:
                np.testing.assert_array_equal(z['target'],np.stack([ds.m128[n] for n in z['names']]))
                np.testing.assert_allclose(m.cv.model.metrics(z['probability'],z['target'],r['threshold'])['per_image'],r['selected']['per_image'],atol=1e-12)
            original=r['diagnostics']
            assert m.diagnose(r)['diagnostics']==original
        if v['passed']:
            for method in m.cv.METHODS:
                names=[n for r in rows if r['method']==method for n in r['val_names']]
                assert len(names)==len(set(names))==170
        print(variant,'verified',len(rows),'fits, metrics, split coverage and diagnostics')


if __name__=='__main__': main()
