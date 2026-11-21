"""Measure synthetic-heldout retention after real finetuning; never selects a model."""
import json
import importlib.util
from pathlib import Path
import numpy as np
import torch
spec = importlib.util.spec_from_file_location('synthetic_training_runner', Path(__file__).with_name('run.py'))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

def main():
    run.DEVICE = torch.device('cuda:1')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    chunks = [torch.load(run.OUT / 'shards' / f'{i:03d}.pt', weights_only=True) for i in (20, 21)]
    features = [torch.cat([c['features'][level] for c in chunks]).to(run.DEVICE) for level in range(4)]
    names = sum([c['names'] for c in chunks], [])
    target = torch.cat([c['target'] for c in chunks]).numpy()
    for variant in ('control', 'synthetic'):
        for seed in (42, 7, 123):
            for fold in (1, 2, 3):
                stem = f'{variant}_fold{fold}_seed{seed}'
                destination = run.OUT / 'cache' / f'{stem}_synthetic_holdout.npz'
                if destination.exists(): continue
                ck = torch.load(run.OUT / f'{stem}.pt', weights_only=True)
                head, _, _ = run.initialize(features, seed, ck['state_dict'])
                p = run.predict(head, features, np.arange(200))
                np.savez_compressed(destination, names=names, probability=p, target=target, threshold=ck['threshold'])
                print('RETENTION', stem, run.cv.model.metrics(p, target, .5)['miou'], flush=True)

if __name__ == '__main__': main()
