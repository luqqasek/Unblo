"""Meaningful edge-case, geometry, artifact, and source-preservation checks."""
from common import *
from execution_audit import inverse_grid,inverse_feature_grid
from metrics_audit import one,geometry
import pandas as pd

def main():
    empty=np.zeros((128,128),bool);full=~empty
    assert official.miou_image(empty,empty)==official.miou_image(full,full)==1
    assert official.miou_image(empty,full)==official.miou_image(full,empty)==0
    g=empty.copy();g[20:28,40:48]=1
    assert abs(official.miou_image(empty,g)-(1-g.mean())/2)<1e-12
    row=one(np.full(g.shape,.5,dtype=np.float32),g,.5)
    assert row['roc_auc']==.5 and abs(row['ap']-g.mean())<1e-12
    assert np.isnan(one(np.zeros_like(g,dtype=np.float32),empty,.5)['roc_auc'])
    assert np.isnan(one(np.zeros_like(g,dtype=np.float32),empty,.5)['foreground_precision'])
    assert geometry(g,g)['boundary_f1_1']==1
    assert geometry(g,empty)['boundary_f1_1']==0
    neutral=np.array([[399/398,0,1/398],[0,399/398,1/398]])
    for n in (13,25,32,128):
        y,x=torch.meshgrid((torch.arange(n)+.5)*2/n-1,(torch.arange(n)+.5)*2/n-1,indexing='ij')
        np.testing.assert_allclose(inverse_grid(neutral,n,n),torch.stack([x,y],-1)[None],atol=1e-7)
    for level,n in enumerate((32,32,25,13)):
        y,x=torch.meshgrid((torch.arange(n)+.5)*2/n-1,(torch.arange(n)+.5)*2/n-1,indexing='ij')
        np.testing.assert_allclose(inverse_feature_grid(neutral,level,n),torch.stack([x,y],-1)[None],atol=1e-7)
    df=pd.read_csv(OUT/'per_image_metrics.csv');summary=pd.read_csv(OUT/'summary_metrics.csv')
    assert len(df)==1020 and len(summary)==6
    for keys,g in df.groupby(['watermark_method','variant','seed']):
        assert len(g)==g.image_id.nunique()==170
        s=summary[(summary.watermark_method==keys[0])&(summary.variant==keys[1])&(summary.seed==keys[2])].iloc[0]
        assert abs(g.official_miou.mean()-s.official_miou_mean)<1e-12
        assert np.all(g.oracle_threshold_miou>=g.official_miou-1e-7)
        assert np.allclose(g.delta_vs_empty,g.official_miou-g.empty_baseline_miou)
    manifest=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
    for path,digest in manifest['sha256'].items():assert sha(path)==digest,path
    results=json.loads((OUT/'diagnostic_results.json').read_text(encoding='utf-8'))
    for ck in results['checkpoints']:assert sha(ROOT/ck['path'])==ck['sha256']
    images=list((OUT/'examples').glob('*.jpg'));assert len(images)==510
    assert len(json.loads((OUT/'examples/index.json').read_text(encoding='utf-8')))==510
    probes=pd.read_csv(OUT/'feature_probe_results.csv');assert len(probes)==28
    confirmation=pd.read_csv(OUT/'alignment_probe_per_image.csv')
    pair=json.loads((OUT/'alignment_probe_confirmation.json').read_text())
    raw=confirmation[confirmation.variant=='all'].set_index('image_id')
    aligned=confirmation[confirmation.variant=='all_aligned'].set_index('image_id').loc[raw.index]
    assert len(raw)==len(aligned)==170 and raw.index.nunique()==170
    for metric in ('miou','fg_iou','auc','ap'):
        assert abs(raw[metric].mean()-pair['summary'][metric]['raw'])<1e-12
        assert abs(aligned[metric].mean()-pair['summary'][metric]['aligned'])<1e-12
        assert abs((aligned[metric]-raw[metric]).mean()-pair['summary'][metric]['delta'])<1e-12
    save_json(OUT/'verification.json',dict(status='passed',per_image_rows=1020,unique_images=510,cohorts=6,probe_runs=28,examples=510,
        checks=['official metric empty union and inverse masks','constant prediction AUC/AP','undefined values','boundary empty cases','exact inverse STN neutral geometry','native conv and pooling centers','summary reconciliation','oracle monotonicity','all recorded source/checkpoint hashes unchanged','example coverage']))
    print('PASS: metric/geometry edge cases, source hashes, 1020 rows / 510 unique images, 28 probes, 510 examples.')

if __name__=='__main__':main()
