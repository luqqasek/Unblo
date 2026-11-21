"""Meaningful checks: official metrics, tie handling, splits, artifacts, source preservation."""
import itertools
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from models import Model,prepare_input,cv
from evaluate import image_metrics,topk,aggregate
OUT=Path(__file__).resolve().parent/'results'
ROOT=OUT.parents[2]


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('competition_metric',ROOT/'starter-kit/task-1/wmloc/utils.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    # Independent exact enumeration verifies finite-sample random AP and tie policy.
    g=np.array([[1,0,1],[0,0,0]],dtype=bool)
    expected=image_metrics(np.zeros_like(g,dtype=float),g,.5)
    aps=[average_precision_score(g.ravel(),np.array(p)) for p in itertools.permutations(range(6))]
    assert abs(np.mean(aps)-expected['random_ap'])<1e-12
    assert abs(topk(np.ones_like(g,dtype=float),g)-g.mean())<1e-12
    rng=np.random.default_rng(1)
    for target in [np.zeros((8,8),bool),np.ones((8,8),bool),rng.random((8,8))>.8]:
        p=rng.random((8,8));m=image_metrics(p,target,.4)
        assert abs(m['miou']-official.miou_image(p>=.4,target))<1e-12
    x=prepare_input(torch.ones(1,3,512,512))
    assert x.shape==(1,12,256,256) and x[:,3:].abs().max()<1e-6
    # A is exactly the imported baseline; D leaves its entire Head intact.
    channels=[32,64,128,128]
    torch.manual_seed(42);a=Model('A',channels)
    torch.manual_seed(42);d=Model('D',channels)
    assert all(torch.equal(v,d.head.state_dict()[k]) for k,v in a.head.state_dict().items())
    features=[torch.rand(2,c,32,32) for c in channels]
    assert torch.equal(a(features,None),a.head(features))
    for v in 'ABCD':
        m=Model(v,channels)
        output=m(features,torch.rand(2,12,256,256))
        assert output.shape==(2,128,128) and torch.isfinite(output).all()
    config=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
    assert all(digest(ROOT/p)==h for p,h in config['sources_sha256'].items())
    config_hash=digest(OUT/'config.json')
    splits=json.loads((OUT/'splits.json').read_text())
    counts={};max_reproduction=0.;run_count=0
    # The GPU port must also reproduce historical CPU checkpoint predictions.
    historical_errors={}
    for method in ('pixelseal','trustmark','stegastamp'):
        if method=='stegastamp':
            source=OUT.parent.parent/'stegastamp_stn_alignment/results'
            ck=torch.load(source/'stegastamp_fold1_seed42.pt',weights_only=True)
            z=np.load(source/'stegastamp_fold1_seed42.npz')
            cache=torch.load(source/'aligned_features.pt',weights_only=True)
        else:
            ck=torch.load(cv.OUT/f'{method}_fold1_head.pt',weights_only=True)
            z=np.load(cv.OUT/f'{method}_oof.npz')
            cache=torch.load(cv.model.OUT/f'{method}_features.pt',weights_only=True)
        names=z['names'].tolist()[:8];ids=[cache['names'].index(n) for n in names]
        model=cv.model.Head(ck['channels']).cuda().eval();model.load_state_dict(ck['state_dict'])
        # PS/TM OOF names are not fold-ordered: select actual fold1 images.
        if method!='stegastamp':
            names=splits[method][0]['validation'][:8];ids=[cache['names'].index(n) for n in names]
            expected=z['probability'][[z['names'].tolist().index(n) for n in names]]
        else:expected=z['probability'][:8]
        with torch.inference_mode():p=model([f[ids].cuda() for f in cache['features']]).sigmoid().cpu().numpy()
        error=float(np.max(np.abs(p-expected)));assert error<2e-5,(method,error)
        historical_errors[method]=error
        del cache,model
    for path in sorted(OUT.glob('*_fold*_seed*.json')):
        r=json.loads(path.read_text());run_count+=1
        assert r['config_sha256']==config_hash
        z=np.load(path.with_suffix('.npz'))
        ck=torch.load(path.with_suffix('.pt'),weights_only=True)
        split=splits[r['method']][r['fold']-1]
        assert z['names'].tolist()==ck['validation_names']==split['validation']
        assert ck['train_names']==split['train']
        assert not set(ck['train_names'])&set(ck['validation_names'])
        np.testing.assert_array_equal(z['mask'],z['probability']>=r['threshold'])
        actual=np.mean([official.miou_image(p,g) for p,g in zip(z['mask'],z['target'])])
        assert abs(actual-r['metrics']['miou'])<1e-12
        assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold']
        assert r['epoch']==np.argmax(r['inner_history'])+1
        assert r['threshold']==max(r['inner_threshold_scores'],key=lambda x:x['miou'])['threshold']
        for p,g,row in zip(z['probability'],z['target'],r['per_image']):
            measured=image_metrics(p,g,r['threshold'])
            assert all(row[k]==v for k,v in measured.items())
        assert aggregate(r['per_image'])==r['metrics']
        max_reproduction=max(max_reproduction,r['checkpoint_reproduction_max_error'])
        counts.setdefault((r['method'],r['variant'],r['seed']),[]).extend(z['names'].tolist())
    for (method,variant,seed),names in counts.items():
        assert len(names)==len(set(names))
        promoted=variant in ('A','D') or any(g['method']==method and g['variant']==variant and g['passed']
            for g in json.loads((OUT/'screening.json').read_text()))
        assert len(names)==(170 if promoted else 57)
    done=json.loads((OUT/'completion.json').read_text())
    assert done['expected_runs']==run_count
    verification=dict(status='passed',runs=run_count,source_hashes_unchanged=True,
                      official_metric_match=True,random_ap_enumeration=True,tie_neutral_topk=True,
                      splits_and_oof_coverage=True,selection_inner_only=True,constant_residual_check=True,
                      identical_A_D_head_initialization=True,checkpoint_max_error=max_reproduction,
                      historical_cpu_checkpoints_on_gpu_max_error=historical_errors,
                      evaluation_sha256=digest(Path(__file__).with_name('evaluate.py')),
                      report_sha256=digest(Path(__file__).with_name('report.py')))
    save(OUT/'verification.json',verification)
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
