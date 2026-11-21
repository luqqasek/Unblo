"""Refit heads on all train images, then predict test; upload is a separate command."""
import hashlib
import json
import sys
import time

import numpy as np
import torch

import cross_validate as cv

sys.path.insert(0, str(cv.model.ROOT / 'src'))
from watermark_localization.submission import save_submission, validate_submission

OUT = cv.HERE / 'results' / 'full_train'


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train = cv.model.LocalisationDataset(cv.model.ROOT / 'data', 'train')
    cv_report = json.loads((cv.OUT / 'metrics.json').read_text())
    summary = dict(protocol='Fresh full-train heads; median CV epochs and thresholds; frozen backbones',
                   config=cv.model.CONFIG, methods={})
    # Finish every training fit before opening the test dataset.
    for method in cv.METHODS:
        cache = cv.model.extract(method, train)
        names = cache['names']
        assert len(names) == len(set(names)) == 170
        assert set(names) == {m['image_name'] for m in train.meta if m['method'] == method}
        targets = torch.from_numpy(np.stack([train.m128[n] for n in names])).float()
        folds = cv_report['methods'][method]['folds']
        epochs = int(np.median([f['epoch'] for f in folds]))
        threshold = float(np.median([f['threshold'] for f in folds]))
        head, optimizer, rng = cv.initialize(cache['features'])
        for epoch in range(epochs):
            cv.train_epoch(head, optimizer, rng, cache['features'], targets, np.arange(len(names)))
            print(f'{method} full-train epoch {epoch + 1}/{epochs}', flush=True)
        path = OUT / f'{method}_head.pt'
        torch.save(dict(state_dict=head.state_dict(), channels=[f.shape[1] for f in cache['features']],
                        epoch=epochs, threshold=threshold, train_names=names, config=cv.model.CONFIG), path)
        summary['methods'][method] = dict(train_count=len(names), epochs=epochs, threshold=threshold,
                                          checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        del cache, head, optimizer, targets
    (OUT / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    test = cv.model.LocalisationDataset(cv.model.ROOT / 'data', 'test')
    assert not test.train
    names = [m['image_name'] for m in test.meta]
    assert len(names) == len(set(names)) == 1020
    assert not set(names) & {m['image_name'] for m in train.meta}
    lookup = {n: i for i, n in enumerate(names)}
    masks = np.zeros((len(names), 128, 128), dtype=np.uint8)
    seen = np.zeros(len(names), dtype=int)
    cache_dir = OUT / 'test_features'
    cache_dir.mkdir(exist_ok=True)
    for method in cv.METHODS:
        original_out = cv.model.OUT
        try:
            cv.model.OUT = cache_dir
            cache = cv.model.extract(method, test)
        finally:
            cv.model.OUT = original_out
        assert len(cache['names']) == len(set(cache['names'])) == 340
        assert set(cache['names']) == {m['image_name'] for m in test.meta if m['method'] == method}
        checkpoint = torch.load(OUT / f'{method}_head.pt', weights_only=True)
        head = cv.model.Head(checkpoint['channels'])
        head.load_state_dict(checkpoint['state_dict'])
        prob = cv.predict(head, cache['features'], np.arange(len(cache['names'])))
        assert np.isfinite(prob).all() and np.all((prob >= 0) & (prob <= 1))
        prediction = (prob >= checkpoint['threshold']).astype(np.uint8)
        ids = [lookup[n] for n in cache['names']]
        masks[ids] = prediction
        seen[ids] += 1
        np.savez_compressed(OUT / f'{method}_test.npz', probability=prob, names=cache['names'],
                            threshold=checkpoint['threshold'])
        summary['methods'][method].update(test_count=len(ids), watermark_fraction=float(prediction.mean()))
        print(f'{method}: {len(ids)} test predictions complete', flush=True)
        del cache, head, prob
    assert np.all(seen == 1)
    path = save_submission(OUT / 'submission.pt', names, torch.from_numpy(masks), names)
    saved = torch.load(path, weights_only=True)
    validate_submission(saved, names)
    # Verify name-based assembly independently against saved per-method probabilities.
    for method in cv.METHODS:
        with np.load(OUT / f'{method}_test.npz') as data:
            ids = [lookup[n] for n in data['names'].tolist()]
            np.testing.assert_array_equal(saved['masks'][ids].numpy(), data['probability'] >= data['threshold'])
    summary.update(test_count=len(names), bytes=path.stat().st_size,
                   submission_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   seconds=time.perf_counter() - start, validation='Passed: format, complete coverage, name mapping, thresholded probabilities')
    (OUT / 'prediction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
