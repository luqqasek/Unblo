"""Shared immutable sources and independent output directory for scaling."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results'
OLD = HERE.parent / 'synthetic_pretrain_2k'
METHODS = ('pixelseal', 'trustmark', 'stegastamp')
VARIANTS = ('real_only', 'n2k_s2500', 'n4k_s2500', 'n2k_s5000', 'n4k_s5000')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def save(path, data):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def train_ids(n):
    return np.arange(2000) if n == 2000 else np.r_[np.arange(2000), np.arange(2200, 4200)]

def real_source(method):
    return (HERE.parent / 'stegastamp_stn_alignment/results/aligned_features.pt' if method == 'stegastamp'
            else HERE.parent / f'frozen_extractor/results/{method}_features.pt')

def splits(method):
    if method == 'stegastamp':
        return json.loads((HERE.parent / 'stegastamp_stn_alignment/results/splits.json').read_text())
    rows = json.loads((HERE.parent / 'frozen_extractor/results/cv3/folds.json').read_text())[method]
    return [dict(fold=i + 1, **r) for i, r in enumerate(rows)]

def setup(device):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device.startswith('cuda'): torch.cuda.set_device(torch.device(device))
