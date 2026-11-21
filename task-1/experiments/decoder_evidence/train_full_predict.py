"""Refit message-aware StegaStamp on all train, then build full test submission."""
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import torch
from PIL import Image
from prepare import HERE,OUT as EXP_OUT,cv,prepare,prepare_bits,occlusion_logits,occlusion_bits

spec=importlib.util.spec_from_file_location('evidence_run',HERE/'run.py')
experiment=importlib.util.module_from_spec(spec);spec.loader.exec_module(experiment)
sys.path.insert(0,str(cv.model.ROOT/'src'))
from watermark_localization.submission import save_submission,validate_submission

OUT=EXP_OUT/'full_train'
PREVIOUS=HERE.parent/'stegastamp_stable_selection/results/full_train'
TEST_FEATURES=cv.HERE/'results/full_train/test_features'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4); OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    source=EXP_OUT/'stable_summary.json'; report=json.loads(source.read_text())
    assert report['status']=='complete' and len(report['runs'])==3
    epochs=int(np.median([r['epoch'] for r in report['runs']]))
    threshold=float(np.median([r['threshold'] for r in report['runs']]))
    train=cv.model.LocalisationDataset(cv.model.ROOT/'data','train')
    cache=cv.model.extract('stegastamp',train)
    evidence=prepare_bits('stegastamp',prepare('stegastamp'))
    names=cache['names']; assert names==evidence['names'] and len(set(names))==170
    assert set(names)=={r['image_name'] for r in train.meta if r['method']=='stegastamp'}
    features=cache['features']+[evidence['perbit']]
    y=torch.from_numpy(np.stack([train.m128[n] for n in names])).float()
    channels=[f.shape[1] for f in cache['features']]
    torch.manual_seed(42); head=experiment.EvidenceHead(channels,'bit_residual',100)
    checkpoint=OUT/'stegastamp_head.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=True)
        assert saved['train_names']==names and saved['epoch']==epochs and saved['threshold']==threshold and saved['source_cv_sha256']==digest(source)
        head.load_state_dict(saved['state_dict']);print('Resuming saved full-train head',flush=True)
    else:
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(42)
        for epoch in range(epochs):
            cv.train_epoch(head,optimizer,rng,features,y,np.arange(170))
            print('Full train',epoch+1,'/',epochs,flush=True)
        torch.save(dict(state_dict=head.state_dict(),channels=channels,evidence_channels=100,variant='bit_residual',
            epoch=epochs,threshold=threshold,seed=42,train_names=names,source_cv_sha256=digest(source)),checkpoint)
    training_summary=dict(epochs=epochs,threshold=threshold,seed=42,train_count=170,source_cv_sha256=digest(source),checkpoint_sha256=digest(checkpoint))
    (OUT/'training_summary.json').write_text(json.dumps(training_summary,indent=2),encoding='utf-8')
    del features,cache,evidence,y
    # All choices fixed before opening test data.
    test=cv.model.LocalisationDataset(cv.model.ROOT/'data','test');assert not test.train
    expected=[r['image_name'] for r in test.meta];assert len(expected)==len(set(expected))==1020
    assert not set(expected)&{r['image_name'] for r in train.meta}
    previous=torch.load(PREVIOUS/'submission.pt',weights_only=True);validate_submission(previous,expected)
    cache=cv.model.extract('stegastamp',test,cache_dir=TEST_FEATURES)
    names=cache['names'];assert len(names)==len(set(names))==340
    meta={r['image_name']:r for r in test.meta}
    assert set(names)=={r['image_name'] for r in test.meta if r['method']=='stegastamp'}
    wm=cv.model.load_watermark('stegastamp',cv.model.ROOT/'checkpoints','cpu')
    maps=[];accuracies=[]
    with torch.inference_mode():
        for i,name in enumerate(names):
            path=OUT/'test_occlusion'/f'{name}.pt'
            if path.exists():logits=torch.load(path,weights_only=True)
            else:
                rgb=np.asarray(Image.open(cv.model.ROOT/'data/test/images'/f'{name}.png').convert('RGB'))
                logits=occlusion_logits(wm,cv.model.to_tensor(rgb)[0])
                path.parent.mkdir(parents=True,exist_ok=True);torch.save(logits,path)
            message=torch.tensor([int(c) for c in meta[name]['message']],dtype=torch.float32)
            maps.append(occlusion_bits(logits,message))
            accuracies.append(float(((logits[0]>0)==message).float().mean()))
            if (i+1)%10==0:print('Test message evidence',i+1,'/340',flush=True)
    maps=torch.stack(maps);assert maps.shape==(340,100,4,4) and torch.isfinite(maps).all()
    torch.save(dict(names=names,perbit=maps),OUT/'stegastamp_test_evidence.pt')
    features=cache['features']+[maps]
    prob=cv.predict(head,features,np.arange(340));assert np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1))
    restored=experiment.EvidenceHead(channels,'bit_residual',100)
    restored.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict'])
    np.testing.assert_allclose(cv.predict(restored,features,np.arange(340)),prob,atol=1e-7)
    lookup={n:i for i,n in enumerate(previous['image_names'])};ids=[lookup[n] for n in names]
    masks=previous['masks'].clone();masks[ids]=torch.from_numpy((prob>=threshold).astype(np.uint8))
    path=save_submission(OUT/'submission.pt',previous['image_names'],masks,expected)
    saved=torch.load(path,weights_only=True);validate_submission(saved,expected)
    other=[i for i in range(1020) if i not in set(ids)]
    assert len(other)==680 and torch.equal(saved['masks'][other],previous['masks'][other])
    np.testing.assert_array_equal(saved['masks'][ids].numpy(),prob>=threshold)
    np.savez_compressed(OUT/'stegastamp_test.npz',probability=prob,names=names,threshold=threshold)
    summary=dict(**training_summary,test_count=1020,updated_stegastamp_images=340,unchanged_other_images=680,
        bytes=path.stat().st_size,previous_submission_sha256=digest(PREVIOUS/'submission.pt'),submission_sha256=digest(path),
        watermark_fraction=float((prob>=threshold).mean()),global_bit_accuracy=float(np.mean(accuracies)),seconds=time.perf_counter()-start,
        validation='Names, binary masks, shape, size, unchanged methods and reloaded checkpoint predictions verified')
    (OUT/'prediction_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
