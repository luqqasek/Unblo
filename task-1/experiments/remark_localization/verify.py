"""Verify official metrics, coverage, calibration separation, controls, and immutable sources."""
import pandas as pd
from evaluate import curve
from common import *


def main():
    provenance=json.loads((OUT/'provenance.json').read_text())
    assert provenance['version']=='0.8.0' and provenance['device'].startswith('cuda')
    assert provenance['pretrained_weights_frozen'] and not provenance['model_training']
    for path,expected in provenance['sources'].items(): assert sha(path)==expected,path
    from trustmark.trustmark import MODEL_CHECKSUMS
    for path in provenance['sources']:
        p=Path(path)
        if p.name in MODEL_CHECKSUMS:
            assert hashlib.md5(p.read_bytes()).hexdigest()==MODEL_CHECKSUMS[p.name]
    assert sha(HERE/'common.py')==provenance['common_sha256']
    assert sha(HERE/'extract.py')==provenance['script_sha256']
    names,gt,base,baset,_=real_data()
    official_gt=np.load(ROOT/'data/train/masks_128.npz')
    gt_by_name=dict(zip(official_gt['names'].tolist(),official_gt['masks']))
    for name,g in zip(names,gt):np.testing.assert_array_equal(g,gt_by_name[name])
    df=pd.read_csv(OUT/'per_image_metrics.csv',float_precision='round_trip')
    assert df.groupby('method').size().eq(170).all() and df.image_id.nunique()==170
    assert not df.duplicated(['image_id','method']).any()
    cal=json.loads((OUT/'calibration.json').read_text())
    original=json.loads((BASE/'folds.json').read_text())['trustmark']
    for record in cal['maps']:
        split=original[record['fold']-1]
        assert record['calibration_names']==split['inner_validation']
        assert record['validation_names']==split['validation']
        assert not set(record['calibration_names'])&set(record['validation_names'])
    for record in cal['family_selection']:
        choices=[r for r in cal['maps'] if r['fold']==record['fold'] and (
            (record['family']=='B_full_selected' and r['method'].startswith(('R1_','R2_','R3_','R4_'))) or
            (record['family']=='R5_selected' and r['method'].startswith('R5_')) or
            (record['family']=='generic_selected' and r['method'].startswith('control_')) or
            (record['family']=='C_local_selected' and r['method'].startswith('local')))]
        assert record['calibration_miou']==max(r['calibration_miou'] for r in choices)
    replayed=0
    for i,name in enumerate(names):
        maps={}
        for folder in ('full','local'):
            with np.load(OUT/folder/f'{name}.npz') as z: maps.update({k:z[k] for k in z.files})
        assert len(maps)==51 and all(v.shape==(128,128) and np.isfinite(v).all() for v in maps.values())
        rec=Image.open(OUT/'recovered'/f'{name}.png')
        assert rec.mode=='RGB' and rec.size==(512,512)
        for _,row in df[df.image_id==name].iterrows():
            if row.method=='baseline':score=base[i]
            elif row.method in maps:score=maps[row.method]
            else:
                choice=next(r for r in cal['family_selection'] if r['family']==row.method and r['fold']==int(row.fold))
                score=maps[choice['method']]
            p=score>=float(row.threshold)
            measured=official.miou_image(p,gt[i])
            assert abs(measured-row.miou)<1e-12,(name,row.method)
            assert abs(curve(score,gt[i],[float(row.threshold)])[0]-measured)<1e-12
            assert row.oracle_miou+1e-12>=max(official.miou_image(np.zeros_like(p),gt[i]),official.miou_image(np.ones_like(p),gt[i]))
            replayed+=1
    # Exact-threshold behavior including float32 rounding and empty/full class conventions.
    for g in (np.zeros((4,4),bool),np.ones((4,4),bool),np.eye(4,dtype=bool)):
        s=np.linspace(0,1,16,dtype=np.float32).reshape(4,4)
        for threshold in (-1,.3,float(s[1,1])+1e-9,1,2):
            assert abs(curve(s,g,[threshold])[0]-official.miou_image(s>=threshold,g))<1e-12
    counts=pd.read_csv(OUT/'summary_by_area.csv')
    for method,group in df.groupby('method'):
        summary=counts[(counts.method==method)&(counts.stratum=='all')].iloc[0]
        for key in ('miou','auc','ap','topk_precision','fg_iou','clean_iou'):
            assert abs(group[key].mean()-summary[key])<1e-12
    sources=json.loads((OUT/'controls/sources.json').read_text())
    assert len(sources['images'])==len({r['processed_rgb_sha256'] for r in sources['images']})==24
    # Ensure diagnostic clean covers are not exact copies of real competition training images.
    cover_hashes={r['processed_rgb_sha256'] for r in sources['images']}
    real_hashes={hashlib.sha256(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest() for p in (ROOT/'data/train/images').glob('*.png')}
    assert not cover_hashes&real_hashes
    pairs=pd.read_csv(OUT/'controls/matched_pairs.csv')
    assert len(pairs)==144 and pairs.image_id.nunique()==24
    alignment=pd.read_csv(OUT/'controls/alignment.csv')
    assert alignment.batch_api_max_pixel_error.max()<=1
    examples=json.loads((OUT/'examples/index.json').read_text())
    assert len(examples)>=20 and len({r['image_id'] for r in examples})==len(examples)
    assert all((OUT/'examples'/r['file']).exists() for r in examples)
    gate=json.loads((OUT/'integration_gate.json').read_text())
    assert gate['passed']==any(r['passed'] for r in gate['variants'])
    assert not list(OUT.rglob('*.pt')), 'Frozen evidence stage should not create learned heads.'
    result=dict(status='passed',unique_real_images=170,score_maps_per_image=51,official_miou_replayed=replayed,
                calibration_split_separation=True,summary_reconciled=True,exact_threshold_ties=True,
                original_model_and_data_hashes_unchanged=True,clean_photos=24,matched_controls=144,
                clean_source_exact_duplicates_with_train=0,visual_examples=len(examples),gpu=provenance['gpu'],
                batch_api_max_uint8_difference=int(alignment.batch_api_max_pixel_error.max()),integration_gate_passed=gate['passed'])
    save(OUT/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
