"""User-authorized full real-data finetuning and test inference; upload stays separate."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results/full_train'
PREVIOUS = HERE.parent / 'stegastamp_stable_selection/results/full_train'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def runner():
    spec = importlib.util.spec_from_file_location('synthetic_deploy_runner', HERE / 'run.py')
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    run.DEVICE = torch.device('cuda:1')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return run

def fit():
    run = runner()
    reports = [json.loads((HERE / f'results/synthetic_fold{f}_seed42.json').read_text()) for f in (1, 2, 3)]
    epochs = int(np.median([r['epoch'] for r in reports]))
    threshold = float(np.median([r['threshold'] for r in reports]))
    source = HERE / 'results/pretrain_seed42.pt'
    protocol = dict(seed=42, epochs=epochs, threshold=threshold,
                    selection='User selected synthetic→real. Default seed42; median epoch/threshold from its three CV folds; no test tuning.',
                    pretrain_sha256=sha(source), aligned_real_features_sha256=sha(run.BASE / 'aligned_features.pt'),
                    cv_sha256=[sha(HERE / f'results/synthetic_fold{f}_seed42.json') for f in (1, 2, 3)],
                    code_sha256=sha(__file__), torch=str(torch.__version__), device='cuda:1',
                    device_name=torch.cuda.get_device_name(1), geometry='inverse actual STN',
                    optimizer='AdamW lr=0.001 weight_decay=0.0001; reset after pretrain', batch_size=8)
    if (OUT / 'training_summary.json').exists():
        old = json.loads((OUT / 'training_summary.json').read_text())
        assert old['protocol'] == protocol and old['checkpoint_sha256'] == sha(OUT / 'stegastamp_head.pt')
        print('FULL FIT ALREADY VERIFIED', flush=True); return
    names, features, gt, _ = run.real_data()
    assert len(set(names)) == 170
    pretrained = torch.load(source, weights_only=True)
    head, opt, rng = run.initialize(features, 42, pretrained['state_dict'])
    targets = torch.from_numpy(gt).float().to(run.DEVICE)
    start = time.perf_counter(); history = []
    for epoch in range(epochs):
        loss = run.train_epoch(head, opt, rng, features, targets, np.arange(170))
        history.append(loss)
        print(f'FULL REAL FINETUNE {epoch + 1}/{epochs} loss={loss:.6f}', flush=True)
    state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    ck = dict(state_dict=state, channels=[f.shape[1] for f in features], seed=42, epoch=epochs,
              threshold=threshold, train_names=names, protocol=protocol)
    torch.save(ck, OUT / 'stegastamp_head.pt')
    save(OUT / 'training_summary.json', dict(protocol=protocol, real_train_count=170, synthetic_train_count=2000,
         checkpoint_sha256=sha(OUT / 'stegastamp_head.pt'), losses=history, seconds=time.perf_counter() - start))
    assert sha(source) == protocol['pretrain_sha256']

def extract():
    assert (OUT / 'training_summary.json').exists(), 'Fit and selection must precede test inference.'
    from PIL import Image
    import onnxruntime as ort
    # Same implementation and instrumented graph as synthetic generation.
    spec = importlib.util.spec_from_file_location('synthetic_deploy_prepare', HERE / 'prepare.py')
    prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
    metadata_path = ROOT / 'data/test/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    names = [r['image_name'] for r in metadata if r['method'] == 'stegastamp']
    assert len(names) == len(set(names)) == 340
    graph = HERE / 'results/cache/features.onnx'
    protocol = dict(metadata_sha256=sha(metadata_path), graph_sha256=sha(graph), code_sha256=sha(__file__),
                    alignment_sha256=sha(HERE.parent / 'diagnostic_audit/execution_audit.py'))
    cache_path = OUT / 'aligned_test_features.pt'
    if cache_path.exists():
        c = torch.load(cache_path, weights_only=True)
        assert c['names'] == names and c['protocol'] == protocol
        return
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])
    wm = prep.StegaStamp.__new__(prep.StegaStamp); wm.sess = session; wm.device = 'cpu'
    samples = []; matrices = []; hashes = []
    for i, name in enumerate(names):
        image = np.asarray(Image.open(ROOT / 'data/test/images' / f'{name}.png').convert('RGB'))
        assert image.shape == (512, 512, 3)
        hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
        values = session.run(prep.FEATURE_NAMES + ['gen_stn/Reshape:0'],
                             {'image': wm._prep(prep.to_tensor(image)), 'secret': np.zeros((1, 100), np.float32)})
        theta = values[-1][0]; matrices.append(theta.tolist()); features = []
        for level, value in enumerate(values[:4]):
            f = torch.from_numpy(value)
            f = F.adaptive_avg_pool2d(f, (min(32, f.shape[-2]), min(32, f.shape[-1])))
            f = F.grid_sample(f, prep.inverse_feature_grid(theta, level, f.shape[-1]), mode='bilinear', align_corners=False, padding_mode='border')
            assert torch.isfinite(f).all(); features.append(f[0].clone())
        samples.append(features)
        if (i + 1) % 50 == 0: print('TEST ALIGNED FEATURES', i + 1, '/340', flush=True)
    torch.save(dict(names=names, features=[torch.stack([f[k] for f in samples]) for k in range(4)], protocol=protocol), cache_path)
    save(OUT / 'test_feature_provenance.json', dict(protocol=protocol, names=names, image_sha256=hashes, theta=matrices))

def predict():
    run = runner()
    sys.path.insert(0, str(ROOT / 'src'))
    from watermark_localization.submission import save_submission, validate_submission
    metadata = json.loads((ROOT / 'data/test/metadata.json').read_text())
    expected = [r['image_name'] for r in metadata]
    assert len(expected) == len(set(expected)) == 1020
    prior_path = PREVIOUS / 'submission.pt'
    receipt = json.loads((PREVIOUS / 'upload_receipt.json').read_text())
    assert json.loads(receipt['response'])['submission_id'] == 1202
    assert sha(prior_path) == receipt['sha256']
    previous = torch.load(prior_path, weights_only=True); validate_submission(previous, expected)
    old_sha = sha(prior_path)
    ck = torch.load(OUT / 'stegastamp_head.pt', weights_only=True)
    assert not set(ck['train_names']) & set(expected)
    cache = torch.load(OUT / 'aligned_test_features.pt', weights_only=True)
    assert set(cache['names']) == {r['image_name'] for r in metadata if r['method'] == 'stegastamp'}
    features = [f.to(run.DEVICE) for f in cache['features']]
    head, _, _ = run.initialize(features, 42, ck['state_dict'])
    probability = run.predict(head, features, np.arange(340))
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    restored, _, _ = run.initialize(features, 42, torch.load(OUT / 'stegastamp_head.pt', weights_only=True)['state_dict'])
    np.testing.assert_array_equal(run.predict(restored, features, np.arange(340)), probability)
    lookup = {n: i for i, n in enumerate(previous['image_names'])}
    ids = [lookup[n] for n in cache['names']]
    other = [i for i in range(1020) if i not in set(ids)]
    masks = previous['masks'].clone()
    masks[ids] = torch.from_numpy((probability >= ck['threshold']).astype(np.uint8))
    path = OUT / 'submission.pt'
    assert not path.exists(), 'Do not overwrite an existing submission artifact.'
    save_submission(path, previous['image_names'], masks, expected)
    saved = torch.load(path, weights_only=True); validate_submission(saved, expected)
    assert torch.equal(saved['masks'][other], previous['masks'][other]) and len(other) == 680
    np.testing.assert_array_equal(saved['masks'][ids].numpy(), probability >= ck['threshold'])
    assert sha(prior_path) == old_sha
    np.savez_compressed(OUT / 'stegastamp_test.npz', names=cache['names'], probability=probability, threshold=ck['threshold'])
    summary = dict(model='synthetic pretrain 2000 → real finetune170; inverse STN; seed42', seed=42,
                   epochs=ck['epoch'], threshold=ck['threshold'], test_count=1020, updated_stegastamp_images=340,
                   unchanged_pixelseal_trustmark_images=680, source_submission_id=1202,
                   source_submission_sha256=old_sha, checkpoint_sha256=sha(OUT / 'stegastamp_head.pt'),
                   submission_sha256=sha(path), bytes=path.stat().st_size,
                   watermark_fraction_stegastamp=float(masks[ids].float().mean()),
                   changed_pixels_vs_previous=int((saved['masks'][ids] != previous['masks'][ids]).sum()),
                   validation='PASS names, coverage, binary128x128, size, checkpoint reload, exact assembly, other methods unchanged')
    save(OUT / 'prediction_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['fit', 'extract', 'predict'])
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4)
    {'fit': fit, 'extract': extract, 'predict': predict}[args.stage]()
