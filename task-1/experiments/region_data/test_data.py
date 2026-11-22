import json
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.region_data.data import RegionData

def test_provenance_and_same_edit_twins(tmp_path):
    src=tmp_path/'sources/coco2017';src.mkdir(parents=True)
    base=tmp_path/'encoded/pixelseal/pilot-train';base.mkdir(parents=True)
    extra=tmp_path/'region-data/pixelseal/train';extra.mkdir(parents=True)
    rows=[];sources=[]
    for i in range(4):
        clean=np.full((512,512,3),30+i*40,np.uint8);marked=clean.copy()
        # Zero quantized residual is deliberately allowed for one encoded base.
        if i:marked+=1
        Image.fromarray(clean).save(base/f'{i}c.png');Image.fromarray(marked).save(base/f'{i}m.png')
        rows.append(dict(realization_id=str(i),source_id=str(i),partition='train',method='pixelseal',message=[0]*256,source_crop=[0,0,512,512],rectangle=[0,0,512,512],clean_path=f'{i}c.png',marked_path=f'{i}m.png',regime='large_source_remnant',primary_kind='partial',minority_fraction=.02,clean_minority=False))
        sources.append(dict(source_id=str(i),objects=[dict(id=i,segmentation=[[80,80,120,80,120,120,100,100,80,120]])]))
    (base/'encoded.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(src/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sources))
    row=dict(rows[0],realization_id='extra',clean_path=str(base/'0c.png'),marked_path=str(base/'0m.png'))
    (extra/'extra.jsonl').write_text(json.dumps(row)+'\n')
    data=RegionData(tmp_path,'pixelseal','train',preload=False);rng=np.random.default_rng(42);seen=set();zero_residual_positive=False
    for _ in range(80):
        xs,ys,m=data.sample(rng,novel=True,paired=True)
        assert len(xs)==len(ys)==len(m)==2
        assert xs[0].dtype==np.uint8 and ys[0].shape==(128,128)
        assert set(np.unique(ys[0]))<={0,1}
        assert set(np.unique(xs[0])) <= {30,70,71,110,111,150,151}
        if np.array_equal(xs[0],xs[1]) and ys[0].any() and not ys[1].any():zero_residual_positive=True
        if m[0]['operation']!='legacy':
            assert not ys[1].any();seen.add(m[0]['operation'])
            assert m[0]['source_id'] in ['0','1','2','3']
    assert {'marked_object_transplant','clean_object_addition','same_content_object_remnant','same_content_clean_hole'}<=seen

    assert zero_residual_positive
    a=data.batch(np.random.default_rng(50),32,True);b=data.batch(np.random.default_rng(50),32,True)
    assert a[0].shape==(32,3,512,512) and a[1].shape==(32,128,128) and len(a[2])==8
    assert np.array_equal(a[0].numpy(),b[0].numpy()) and a[3]==b[3]
    from experiments.region_data.pair_data import PositionPairs
    # Fixture has only the first area bin; duplicate its candidates for this geometry test.
    data.bybin[(True,1)]=data.bybin[(True,0)]
    position=PositionPairs(data)
    for _ in range(10):
        xs,ys,meta=position.pair(rng)
        assert ys[0].sum()==ys[1].sum() and not np.array_equal(ys[0],ys[1])
        assert meta[0]['realization_id']==meta[1]['realization_id']
        assert set(np.unique(xs[0])) <= {30,70,71,110,111,150,151}


def test_legacy_fast_pixels_and_masks_identical():
    from experiments.region_data.data import legacy_views
    from experiments.native_localizer.generator import pair_views
    rng=np.random.default_rng(1);clean=rng.integers(0,255,(512,512,3),dtype=np.uint8);marked=clean+np.uint8(1)
    for kind in ['clean','marked','partial']:
        for pair in [None,'clean_twin','move','complement']:
            for seed in [1,2,3]:
                row=dict(rectangle=[0,0,512,512],primary_kind=kind,regime='large_source_remnant',minority_fraction=.013,clean_minority=bool(seed%2))
                a=legacy_views(clean,marked,row,np.random.default_rng(seed),pair)
                b=pair_views(clean,marked,row,np.random.default_rng(seed),pair,simple=False)
                assert all(np.array_equal(x,y) for x,y in zip(a[0]+a[1],b[0]+b[1]))
