"""Verify original outer coverage, inner separation, checkpoints and official mIoU."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('stable_runner',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
cv,OUT=runner.cv,runner.OUT


def main():
    torch.set_num_threads(4)
    splits=json.loads((OUT/'splits.json').read_text())
    original_splits=json.loads((cv.OUT/'folds.json').read_text())['stegastamp']
    cache=torch.load(cv.model.OUT/'stegastamp_features.pt',weights_only=True)
    source=np.load(cv.OUT/'stegastamp_oof.npz')
    data=json.loads((OUT/'metrics.json').read_text())
    for split,original in zip(splits,original_splits):
        assert split['train']==original['train'] and split['validation']==original['validation']
        pooled=sum(split['inner_validation'],[])
        assert len(pooled)==len(set(pooled)) and set(pooled)==set(split['train'])
        assert not set(pooled)&set(split['validation'])
    for r in data['runs']:
        stem=f"stegastamp_fold{r['fold']}_seed{r['seed']}"
        saved=np.load(OUT/f'{stem}.npz')
        assert saved['names'].tolist()==r['names']==splits[r['fold']-1]['validation']
        checkpoint=torch.load(OUT/f'{stem}.pt',weights_only=True)
        head=cv.model.Head(checkpoint['channels']);head.load_state_dict(checkpoint['state_dict'])
        ids=[cache['names'].index(n) for n in r['names']]
        prob=cv.predict(head,cache['features'],ids)
        np.testing.assert_allclose(prob,saved['probability'],atol=1e-7)
        assert abs(cv.model.score(prob>=checkpoint['threshold'],source['target'][ids])-r['miou'])<1e-12
        print(stem,'verified')


if __name__=='__main__':main()
