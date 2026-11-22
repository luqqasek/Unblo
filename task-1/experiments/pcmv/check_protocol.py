"""CPU regressions for scientific correctness: payload signs, cache types, oracle."""
import json,tempfile
from pathlib import Path
import numpy as np
from .features import signed_payload
from .proposals import exact_oracle,metrics
from .fit import load


def main():
    for dtype in (np.uint8,np.int64,np.float32):assert np.array_equal(signed_payload(np.array([0,1,0],dtype)),[-1,1,-1])
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory);(p/'features').mkdir();(p/'native512').mkdir();task={'id':'fixture','kind':'real','role':'validation'};(p/'plan.json').write_text(json.dumps({'tasks':[task]}))
        mask=np.zeros((1,128,128),bool);mask[:,20:30,40:50]=1;np.savez(p/'features/fixture.npz',features=np.zeros((1,1)),wrong=np.zeros((1,1)),base=np.full((128,128),.3),gt=mask[0],masks=np.packbits(mask,axis=-1),gains=np.zeros((1,2)))
        (p/'features/fixture.json').write_text(json.dumps(dict(task,base='checkpoint description',signed_bits_fixed=True,feature_keys=['f'],basic_indices=[0])));np.savez(p/'native512/fixture.npz',features=np.zeros((1,7)),wrong=np.zeros((1,7)))
        d=load(p)[0];assert isinstance(d['base'],np.ndarray) and d['base'].shape==(128,128);assert d['base_description']=='checkpoint description';assert d['features'].shape==(1,8)
    rng=np.random.default_rng(42)
    for trial in range(40):
        b=rng.random((8,8))<.4;y=rng.random((8,8))<.3;m=rng.random((5,8,8))<.1;one=two=metrics(b,y)
        for i in range(5):
            for action in (False,True):
                z=b.copy();z[m[i]]=action;one=max(one,metrics(z,y));two=max(two,one)
                for j in range(5):
                    if (m[i]&m[j]).any():continue
                    for second in (False,True):
                        zz=z.copy();zz[m[j]]=second;two=max(two,metrics(zz,y))
        got=exact_oracle(b,m,y);assert abs(got['one_toggle']-one)<1e-12 and abs(got['two_disjoint_toggles']-two)<1e-12
    print('PASS: unsigned payload signs; numeric cache/metadata separation; exact oracle versus40 brute-force cases')


if __name__=='__main__':main()
