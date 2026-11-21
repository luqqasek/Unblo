"""Recompute diagnostic predictions from all six saved seed checkpoints."""
import json
import numpy as np
import torch
from epoch_probe import cv, OUT


def main():
    torch.set_num_threads(4)
    cache=torch.load(cv.model.OUT/'stegastamp_features.pt',weights_only=True)
    original=np.load(cv.OUT/'stegastamp_oof.npz')
    splits=json.loads((cv.OUT/'folds.json').read_text())['stegastamp']
    data=json.loads((OUT/'metrics.json').read_text())
    for run in data['runs']:
        if run['mode']!='nested':continue
        stem=f"stegastamp_fold{run['fold']}_seed{run['seed']}"
        saved=np.load(OUT/f'{stem}.npz')
        names=saved['names'].tolist()
        assert names==splits[run['fold']-1]['validation']
        ids=[cache['names'].index(n) for n in names]
        checkpoint=torch.load(OUT/f'{stem}.pt',weights_only=True)
        head=cv.model.Head([f.shape[1] for f in cache['features']])
        head.load_state_dict(checkpoint['state_dict'])
        prob=cv.predict(head,cache['features'],ids)
        np.testing.assert_allclose(prob,saved['probability'],atol=1e-7)
        score=cv.model.score(prob>=checkpoint['threshold'],original['target'][ids])
        assert abs(score-run['miou'])<1e-12
        print(stem,'verified',round(score,6))


if __name__=='__main__':main()
