"""Refit seed42 StegaStamp on all train; preserve previous other-method predictions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stable_selection',HERE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
cv=runner.cv
sys.path.insert(0,str(cv.model.ROOT/'src'))
from watermark_localization.submission import save_submission,validate_submission

OUT=HERE/'results/full_train'
PREVIOUS=cv.HERE/'results/full_train'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    measured=json.loads((runner.OUT/'metrics.json').read_text())
    selected=[r for r in measured['runs'] if r['seed']==42]
    assert len(selected)==3
    epochs=int(np.median([r['epoch'] for r in selected]))
    threshold=float(np.median([r['threshold'] for r in selected]))
    train=cv.model.LocalisationDataset(cv.model.ROOT/'data','train')
    cache=cv.model.extract('stegastamp',train)
    names=cache['names']
    assert len(names)==len(set(names))==170
    assert set(names)=={r['image_name'] for r in train.meta if r['method']=='stegastamp'}
    target=torch.from_numpy(np.stack([train.m128[n] for n in names])).float()
    head,opt,rng=runner.initialize(cache['features'],42)
    for epoch in range(epochs):
        cv.train_epoch(head,opt,rng,cache['features'],target,np.arange(len(names)))
        print(f'StegaStamp full train {epoch+1}/{epochs}',flush=True)
    checkpoint=OUT/'stegastamp_head.pt'
    torch.save(dict(state_dict=head.state_dict(),channels=[f.shape[1] for f in cache['features']],
        train_names=names,seed=42,epoch=epochs,threshold=threshold,config=cv.model.CONFIG),checkpoint)
    # Training and hyperparameter selection are finished before loading test data.
    test=cv.model.LocalisationDataset(cv.model.ROOT/'data','test')
    assert not test.train
    expected=[r['image_name'] for r in test.meta]
    assert len(expected)==len(set(expected))==1020
    assert not set(expected)&{r['image_name'] for r in train.meta}
    previous=torch.load(PREVIOUS/'submission.pt',weights_only=True)
    validate_submission(previous,expected)
    lookup={n:i for i,n in enumerate(previous['image_names'])}
    cache=cv.model.extract('stegastamp',test,cache_dir=PREVIOUS/'test_features')
    assert len(cache['names'])==len(set(cache['names']))==340
    assert set(cache['names'])=={r['image_name'] for r in test.meta if r['method']=='stegastamp'}
    prob=cv.predict(head,cache['features'],np.arange(340))
    assert np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1))
    ids=[lookup[n] for n in cache['names']]
    masks=previous['masks'].clone()
    masks[ids]=torch.from_numpy((prob>=threshold).astype(np.uint8))
    path=save_submission(OUT/'submission.pt',previous['image_names'],masks,expected)
    saved=torch.load(path,weights_only=True);validate_submission(saved,expected)
    assert path.stat().st_size<=20_000_000
    np.testing.assert_array_equal(saved['masks'][ids].numpy(),prob>=threshold)
    other=[i for i in range(1020) if i not in set(ids)]
    assert len(other)==680 and torch.equal(saved['masks'][other],previous['masks'][other])
    loaded=torch.load(checkpoint,weights_only=True)
    reloaded=cv.model.Head(loaded['channels']);reloaded.load_state_dict(loaded['state_dict'])
    np.testing.assert_allclose(cv.predict(reloaded,cache['features'],np.arange(340)),prob,atol=1e-7)
    np.savez_compressed(OUT/'stegastamp_test.npz',probability=prob,names=cache['names'],threshold=threshold)
    summary=dict(seed=42,epochs=epochs,threshold=threshold,train_count=170,test_count=1020,
        updated_stegastamp_images=340,unchanged_other_images=680,
        cv_selection='Median of seed42 stratified-inner-CV outer runs; no best-seed selection',
        source_cv_sha256=digest(runner.OUT/'metrics.json'),checkpoint_sha256=digest(checkpoint),
        previous_submission_sha256=digest(PREVIOUS/'submission.pt'),submission_sha256=digest(path),
        bytes=path.stat().st_size,watermark_fraction=float((prob>=threshold).mean()),
        validation='All names, binary shape, size, unchanged methods, and checkpoint predictions verified')
    (OUT/'prediction_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
