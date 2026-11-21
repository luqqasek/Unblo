"""Verify CV coverage, disjoint splits, saved checkpoint predictions and metrics."""
import json

import numpy as np
import torch

import cross_validate as cv


def main():
    torch.set_num_threads(4)
    splits = json.loads((cv.OUT / 'folds.json').read_text())
    report = json.loads((cv.OUT / 'metrics.json').read_text())
    ds = cv.model.LocalisationDataset(cv.model.ROOT / 'data', 'train')
    all_values = []
    for method in cv.METHODS:
        cache = torch.load(cv.model.OUT / f'{method}_features.pt', weights_only=True)
        saved = np.load(cv.OUT / f'{method}_oof.npz')
        names = saved['names'].tolist()
        assert names == cache['names']
        np.testing.assert_array_equal(saved['target'], np.stack([ds.m128[n] for n in names]))
        coverage = []
        for index, split in enumerate(splits[method]):
            train, val = set(split['train']), set(split['validation'])
            assert not train & val
            assert train | val == set(names)
            assert not set(split['inner_train']) & set(split['inner_validation'])
            assert set(split['inner_train']) | set(split['inner_validation']) == train
            coverage.extend(split['validation'])
            checkpoint = torch.load(cv.OUT / f'{method}_fold{index + 1}_head.pt', weights_only=True)
            head = cv.model.Head(checkpoint['channels'])
            head.load_state_dict(checkpoint['state_dict'])
            ids = [names.index(n) for n in split['validation']]
            prob = cv.predict(head, cache['features'], ids)
            np.testing.assert_allclose(prob, saved['probability'][ids], atol=1e-7)
            np.testing.assert_array_equal(saved['threshold'][ids], checkpoint['threshold'])
            metric = cv.model.metrics(prob, saved['target'][ids], checkpoint['threshold'])
            expected = report['methods'][method]['folds'][index]['selected']
            np.testing.assert_allclose(metric['per_image'], expected['per_image'], atol=1e-12)
            all_values.extend(metric['per_image'])
        assert len(coverage) == len(set(coverage)) == len(names)
        oof = cv.model.score(saved['probability'] >= saved['threshold'][:, None, None], saved['target'])
        assert abs(oof - report['methods'][method]['oof_miou']) < 1e-12
        print(method, ': all 3 checkpoints, splits and OOF scores verified')
    assert len(all_values) == len(ds) == 510
    assert abs(np.mean(all_values) - report['overall']['oof_miou']) < 1e-12
    print('Verified all 510 OOF predictions and overall metric.')


if __name__ == '__main__':
    main()
