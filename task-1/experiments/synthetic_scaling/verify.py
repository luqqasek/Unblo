"""Independent checks of generated geometry, provenance, OOF pairing and scores."""
import argparse
import zipfile
from PIL import Image
from scaling_common import *

def data_checks():
    source=module('scaling_verify_generator',OLD/'prepare.py')
    with zipfile.ZipFile(OLD/'results/sources/annotations_trainval2017.zip') as z:
        coco=json.loads(z.read('annotations/instances_val2017.json'))
    annotations={}
    for a in coco['annotations']:annotations.setdefault(a['image_id'],[]).append(a)
    records=sorted(coco['images'],key=lambda r:r['id'])
    order=np.random.default_rng(20261121).permutation(len(records))
    manifests={m:json.loads((OUT/'data'/m/'manifest.json').read_text()) for m in METHODS}
    targets={m:np.load(OUT/'data'/m/'targets.npy',mmap_mode='r') for m in METHODS}
    stats={}
    for m,rows in manifests.items():
        assert len(rows)==len({r['coco_id'] for r in rows})==len({r['cover_sha256'] for r in rows})==4200
        assert [r['coco_id'] for r in rows]==[records[i]['id'] for i in order[:4200]]
        assert all(r['pool_index']==i and r['split']==('validation' if 2000<=i<2200 else 'train') for i,r in enumerate(rows))
        assert all(r['min_real_dhash']>12 and len(r['message'])==(256 if m=='pixelseal' else 100) and set(r['message'])<={'0','1'} for r in rows)
        assert set(train_ids(2000))<set(train_ids(4000))
        assert not set(train_ids(4000))&set(range(2000,2200))
        assert targets[m].shape==(4200,128,128) and targets[m].dtype==np.uint8
        counts={mode:sum(r['mode']==mode for r in rows) for mode in ('clean','full','patch','restore')}
        assert counts==dict(clean=210,full=210,patch=1890,restore=1890)
        features=[]
        for k in range(4):
            a=np.load(OUT/'data'/m/f'features_{k}.npy',mmap_mode='r')
            assert a.dtype==np.float32 and len(a)==4200
            low,high=float('inf'),-float('inf')
            for start in range(0,4200,100):
                v=a[start:start+100];assert np.isfinite(v).all()
                low=min(low,float(v.min()));high=max(high,float(v.max()))
            features.append(dict(shape=list(a.shape),min=low,max=high))
        preflight=json.loads((OUT/'data'/m/'feature_preflight.json').read_text())
        assert len(preflight)==12 and max(r['normalized_rmse'] for r in preflight)<1e-4
        stats[m]=dict(mode_counts=counts,features=features,min_real_dhash=min(r['min_real_dhash'] for r in rows),
                      max_real_feature_normalized_rmse=max(r['normalized_rmse'] for r in preflight),
                      bit_accuracy_by_mode={mode:float(np.mean([r['bit_accuracy'] for r in rows if r['mode']==mode])) for mode in counts})
    with zipfile.ZipFile(OLD/'results/sources/val2017.zip') as archive:
        for i in range(4200):
            record=records[order[i]];raw=archive.read('val2017/'+record['file_name'])
            rng=np.random.default_rng(20261121+i)
            cover,region,shape,crop=source.cover_and_mask(raw,annotations.get(record['id'],[]),rng,['rectangle','ellipse','object'][i%3])
            mode=manifests['pixelseal'][i]['mode']
            mask=np.zeros_like(region) if mode=='clean' else np.ones_like(region) if mode=='full' else region if mode=='patch' else ~region
            gt=source.downsample_mask(mask.astype(np.uint8));cover_sha=hashlib.sha256(cover.tobytes()).hexdigest()
            for m in METHODS:
                r=manifests[m][i]
                assert r['cover_sha256']==cover_sha and r['crop']==crop and r['shape']==shape
                np.testing.assert_array_equal(targets[m][i],gt)
                assert r['area']==float(gt.mean())
                if m=='trustmark' and mode=='patch':
                    x0,y0,x1,y1=r['embedding_bbox'];yy,xx=np.where(region)
                    assert 0<=x0<=xx.min()<=xx.max()<x1<=512 and 0<=y0<=yy.min()<=yy.max()<y1<=512
                    assert max((x1-x0)/(y1-y0),(y1-y0)/(x1-x0))<=2
                example=OUT/'data'/m/'examples'/f'{i:04d}_image.png'
                if example.exists():
                    image=np.asarray(Image.open(example).convert('RGB'))
                    assert hashlib.sha256(image.tobytes()).hexdigest()==r['image_sha256']
                    np.testing.assert_array_equal(image[~mask],cover[~mask])
                    np.testing.assert_array_equal(np.asarray(Image.open(example.with_name(f'{i:04d}_mask.png')))>127,mask)
            if (i+1)%500==0:print('VERIFY GEOMETRY',i+1,'/4200',flush=True)
    ss=[np.load(OUT/'data/stegastamp'/f'features_{k}.npy',mmap_mode='r') for k in range(4)]
    for block in range(22):
        old=torch.load(OLD/f'results/shards/{block:03d}.pt',weights_only=True)
        for arr,v in zip(ss,old['features']):np.testing.assert_array_equal(arr[block*100:(block+1)*100],v.numpy())
        np.testing.assert_array_equal(targets['stegastamp'][block*100:(block+1)*100],old['target'].numpy())
    result=dict(status='passed',unique_cover_geometries=4200,method_masks_verified=12600,prior_ss_features_exact=2200,methods=stats,
                limits='Recomputed all masks and cover hashes; encoded RGB retained only for documented examples. Encoder execution for all samples was checked during generation.')
    save(OUT/'data_verification.json',result)
    return result

def run_checks():
    import pandas as pd
    analysis=module('scaling_verify_metrics',OLD/'report.py')
    rows=pd.read_csv(OUT/'per_image_metrics.csv');assert len(rows)==7650
    summary=json.loads((OUT/'summary.json').read_text())
    canonical=np.load(ROOT/'data/train/masks_128.npz')
    canonical=dict(zip(canonical['names'],canonical['masks']))
    checked=0;max_error=0
    for m in METHODS:
        dest=OUT/'runs'/m;config=json.loads((dest/'config.json').read_text())
        assert config['folds']==splits(m) and config['code_sha256']==sha(HERE/'train.py') and config['common_sha256']==sha(HERE/'scaling_common.py')
        ref=np.load(HERE.parent/f'frozen_extractor/results/cv3/{m}_oof.npz')
        target_by_name=dict(zip(ref['names'],ref['target']))
        for name,target in target_by_name.items():
            np.testing.assert_array_equal(target,canonical[name])
            original=np.asarray(Image.open(ROOT/'data/train/masks'/f'{name}.png'))>127
            np.testing.assert_array_equal(target,analysis.official.downsample_mask(original))
        for seed in (42,7,123):
            for v in VARIANTS[1:]:
                checkpoint=torch.load(dest/f'{v}_seed{seed}_pretrain.pt',weights_only=True)
                n=2000 if v.startswith('n2k') else 4000
                updates=int(v.split('_s')[1])
                assert checkpoint['n']==n and checkpoint['updates']==updates and checkpoint['epoch']==updates*8//n
                assert checkpoint['config_sha256']==sha(dest/'config.json')
                synthetic=np.load(dest/f'{v}_seed{seed}_synthetic_validation.npz')
                np.testing.assert_array_equal(synthetic['target'],np.load(OUT/'data'/m/'targets.npy',mmap_mode='r')[2000:2200])
                transfer=np.load(dest/f'{v}_seed{seed}_synthetic_only.npz')
                assert transfer['names'].tolist()==ref['names'].tolist() and float(transfer['threshold'])==.5
                np.testing.assert_array_equal(transfer['target'],ref['target'])
                assert np.isfinite(transfer['probability']).all()
            if m=='stegastamp':
                reproduction=json.loads((dest/f'prior_reproduction_seed{seed}.json').read_text())
                assert reproduction['max_parameter_error']<1e-6
            for variant in VARIANTS:
                names=[]
                for split in splits(m):
                    stem=f'{variant}_fold{split["fold"]}_seed{seed}'
                    r=json.loads((dest/f'{stem}.json').read_text());ck=torch.load(dest/f'{stem}.pt',weights_only=True)
                    z=np.load(dest/f'{stem}.npz');names+=r['names']
                    assert ck['validation_names']==r['names']==z['names'].tolist()==split['validation']
                    assert ck['train_names']==split['train'] and not set(split['train'])&set(split['validation'])
                    assert ck['config_sha256']==r['config_sha256']==sha(dest/'config.json')
                    assert ck['epoch']==r['epoch']==np.argmax([h['miou'] for h in r['inner_history']])+1
                    assert ck['threshold']==r['threshold']==float(z['threshold'])==max(r['inner_threshold_scores'],key=lambda a:a['miou'])['threshold']
                    assert ck['pretrain_sha256']==(None if variant=='real_only' else sha(dest/f'{variant}_seed{seed}_pretrain.pt'))
                    p,g=z['probability'],z['target'];assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                    if m=='stegastamp' and variant in ('real_only','n2k_s2500'):
                        prior_variant='control' if variant=='real_only' else 'synthetic'
                        prior=np.load(OLD/f'results/{prior_variant}_fold{split["fold"]}_seed{seed}.npz')
                        np.testing.assert_array_equal(p,prior['probability'])
                        assert float(z['threshold'])==float(prior['threshold'])
                    np.testing.assert_array_equal(g,np.stack([target_by_name[n] for n in r['names']]))
                    scores=[analysis.official.miou_image(pp>=r['threshold'],gg) for pp,gg in zip(p,g)]
                    np.testing.assert_allclose(scores,r['per_image'],atol=1e-12)
                    actual=rows[(rows.method==m)&(rows.variant==variant)&(rows.seed==seed)&(rows.fold==split['fold'])].set_index('image_id')
                    np.testing.assert_allclose(actual.loc[r['names'],'miou'],scores,atol=1e-12)
                    max_error=max(max_error,abs(np.mean(scores)-r['miou']));checked+=1
                assert len(names)==len(set(names))==170
        print('VERIFIED OOF',m,flush=True)
    for r in summary['comparisons']:
        actual=rows[(rows.method==r['method'])&(rows.variant==r['variant'])]
        for metric in ('miou','fg_iou','clean_iou','auc','ap','topk'):
            assert abs(actual[metric].mean()-r[metric])<1e-12
    zeros=np.zeros((128,128),bool);half=zeros.copy();half[:64]=True
    assert analysis.measure(zeros.astype(float),zeros,.5)['miou']==1
    assert analysis.measure((~zeros).astype(float),~zeros,.5)['miou']==1
    assert np.isnan(analysis.measure(zeros.astype(float),zeros,.5)['auc'])
    ties=analysis.measure(np.full(half.shape,.5),half,.5)
    assert ties['auc']==ties['ap']==ties['topk']==.5
    return dict(outer_fits_verified=checked,canonical_real_masks_verified=510,prior_stegastamp_oof_runs_bit_exact=18,per_image_rows=len(rows),unique_real_images=rows.image_id.nunique(),max_official_metric_error=max_error,
                metric_edge_cases='passed: empty-union, undefined AUC, tied AP/AUC/Top-K')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2)
    snapshot=json.loads((OUT/'source_snapshot.json').read_text())
    for path,digest in snapshot['files'].items():assert sha(path)==digest,path
    if args.data_only:data_checks();return
    data=json.loads((OUT/'data_verification.json').read_text());assert data['status']=='passed'
    checked=run_checks()
    save(OUT/'verification.json',dict(status='passed',immutable_source_files=len(snapshot['files']),data=data,**checked,
                                     caveat='Repeated development CV; bootstrap intervals do not remove adaptive reuse or multiple-comparison uncertainty. No leaderboard submission.'))
    print('ALL CHECKS PASSED',checked,flush=True)

if __name__=='__main__':main()
