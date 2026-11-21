"""Fail closed on IDs, CV leakage, official-grid alignment and metric mismatches."""
import pandas as pd
from PIL import Image
from common import *

def main():
    records,base,gt,splits,provenance=cohort();names=[r['image_id'] for r in records];index={n:i for i,n in enumerate(names)}
    for src in json.loads((OUT/'inference_trufor.json').read_text())['baseline_sources']:
        assert sha(src['path'])==src['sha256'],'Historical baseline was modified'
    model_count=0;checked=0;max_alignment_error=0.
    for model in ('trufor','mesorch','detectivesam'):
        directory=OUT/model;assert {p.stem for p in directory.glob('*.npz')}==set(names)
        df=pd.read_csv(OUT/f'eval_{model}/per_image_metrics.csv');assert df.groupby('variant').size().eq(510).all()
        calibration=json.loads((OUT/f'eval_{model}/calibration.json').read_text())
        for c in calibration:
            s=splits[c['family']][c['fold']-1]
            assert c['calibration_names']==s['inner_validation'] and c['validation_names']==s['validation']
            assert set(c['calibration_names'])<=set(s['train']) and not set(c['calibration_names'])&set(c['validation_names'])
        for i,n in enumerate(names):
            z=np.load(directory/f'{n}.npz');p=z['probability'];native=z['probability512']
            assert p.shape==(128,128) and native.shape==(512,512) and p.dtype==np.float32
            assert np.isfinite(native).all() and 0<=native.min()<=native.max()<=1
            reference=native.reshape(128,4,128,4).mean((1,3))
            # CPU/GPU float32 reduction order can differ by a few ULPs.
            max_alignment_error=max(max_alignment_error,float(np.max(np.abs(p-reference))))
            np.testing.assert_allclose(p,reference,atol=5e-7,rtol=0)
            assert np.array_equal(z['complement'],~z['binary'])
            expected=p>.5 if model=='detectivesam' else p>=.5
            assert np.array_equal(z['binary'],expected)
            assert np.array_equal(z['native_binary128'],z['binary512'][::4,::4])
            if model=='trufor':assert z['confidence'].shape==(128,128) and np.isfinite(z['confidence']).all()
            if i%17==0:
                r=df[(df.image_id==n)&(df.variant=='direct_fixed')].iloc[0]
                assert abs(official.miou_image(p>=.5,gt[i])-r.miou)<1e-12
                r=df[(df.image_id==n)&(df.variant=='selected')].iloc[0]
                c=next(c for c in calibration if c['family']==r.family and c['fold']==r.fold and c['selected'])
                prob=p if c['orientation']=='direct' else 1-p
                assert abs(official.miou_image(prob>=c['threshold'],gt[i])-r.miou)<1e-12
            checked+=1
        model_count+=1
    for i,n in enumerate(names):
        assert sha(OUT/'inputs/rgb'/f'{n}.png')==sha(ROOT/'data/train/images'/f'{n}.png')
        assert np.array_equal(np.asarray(Image.open(OUT/'inputs/gt'/f'{n}.png'))>127,gt[i])
    # Exhaustive toy cases cover constants/ties and missing classes in our curve.
    for g in [np.zeros((3,3),bool),np.ones((3,3),bool),np.eye(3,dtype=bool)]:
        for p in [np.full((3,3),.5,np.float32),np.arange(9,dtype=np.float32).reshape(3,3)/8]:
            cuts=np.r_[-.1,np.unique(p),1.00001]
            np.testing.assert_allclose(curve(p,g,cuts),[official.miou_image(p>=np.float32(t),g) for t in cuts],atol=1e-12)
    fusion=OUT/'fusion/summary.json';crossfits=0
    if fusion.exists():
        for path in (OUT/'fusion/crossfit').glob('*.json'):
            a=json.loads(path.read_text());tr=set(a['train_names']);va=set(a['validation_names']);bad=set(a['forbidden_names'])
            assert not tr&va and not tr&bad
            it=set(a['epoch_selection_train']);iv=set(a['epoch_selection_validation'])
            assert not it&iv and it|iv==tr and not (it|iv)&va
            crossfits+=1
        assert crossfits==63
        frame=pd.read_csv(OUT/'fusion/per_image_metrics.csv');assert frame.groupby('variant').size().eq(510).all()
        for path in (OUT/'fusion').glob('*_E*.npz'):
            z=np.load(path);config=json.loads(path.with_suffix('.json').read_text())
            assert z['names'].tolist()==config['validation_names']
            assert not set(config['train_names'])&set(config['validation_names'])
            assert not set(config['fusion_fit_names'])&set(config['calibration_names'])
            for n,p in zip(z['names'],z['probability']):
                m=official.miou_image(p>=float(z['threshold']),gt[index[str(n)]])
                saved=frame[(frame.image_id==n)&(frame.variant==config['variant'])].iloc[0]
                assert abs(m-saved.miou)<1e-12
    save(OUT/'verification.json',dict(status='passed',n_images=510,models=model_count,maps_checked=checked,
         official_gt_exact=True,native_alignment_within_float32=True,max_alignment_error=max_alignment_error,baseline_hashes_unchanged=True,crossfit_models_verified=crossfits,
         no_test_images_in_cohort=True,metric_checks='official miou, all/empty/tied scores, calibration and fusion recomputation'))
    print('VERIFIED',checked,'forensic maps;',crossfits,'nested crossfit models;',len(names),'official train images')

if __name__=='__main__':main()
