"""Post-hoc controlled fold-2 probe: seed42, threshold0.5, only training length changes.

18 epochs was selected by seed7's INNER validation; this remains diagnostic.
It must not be reported as a fresh benchmark or used to select on outer labels.
"""
import json
import numpy as np
import torch
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('diagnostic_runner',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
cv,initialize,OUT=runner.cv,runner.initialize,runner.OUT


def main():
    torch.set_num_threads(4)
    cache=torch.load(cv.model.OUT/'stegastamp_features.pt',weights_only=True)
    split=json.loads((cv.OUT/'folds.json').read_text())['stegastamp'][1]
    saved=np.load(cv.OUT/'stegastamp_oof.npz')
    assert cache['names']==saved['names'].tolist()
    ids={k:[cache['names'].index(n) for n in v] for k,v in split.items()}
    targets=torch.from_numpy(saved['target']).float()
    head,opt,rng=initialize(cache['features'],42)
    result=[]
    for epoch in range(1,19):
        cv.train_epoch(head,opt,rng,cache['features'],targets,ids['train'])
        if epoch not in (10,18):continue
        prob=cv.predict(head,cache['features'],ids['validation'])
        if epoch==10:np.testing.assert_allclose(prob,saved['probability'][ids['validation']],atol=1e-7)
        metric=cv.model.metrics(prob,targets[ids['validation']].numpy(),.5)
        result.append(dict(epoch=epoch,miou=metric['miou'],predicted_area=float((prob>=.5).mean())))
    (OUT/'epoch_probe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
