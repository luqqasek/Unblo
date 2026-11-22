import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .data import Pool
from experiments.native_localizer.common import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);a=p.parse_args();result={};ids={}
    for partition in ('train','dev'):
        path=a.root/f'triforensic-data/{a.family}/{partition}';pool=Pool(path);rows=pool.rows
        ids[partition]={r[k] for r in rows for k in ('source_id','background_source_id')}
        geometry=[r for r in rows if r['bucket']==1]
        for r in geometry:
            x,y,x1,y1=r['initial_rectangle'];w=x1-x;h=y1-y
            assert max(w,h)/min(w,h)<=(16 if a.family=='pixelseal' else 2)
        for i in np.linspace(0,len(rows)-1,100,dtype=int):
            m=np.unpackbits(pool.get(i,'support')).reshape(512,512).astype(bool)
            assert np.array_equal(pool.get(i,'rgb')[~m],pool.get(i,'clean')[~m])
            expected=m.reshape(128,4,128,4).sum((1,3))>=8
            assert np.array_equal(expected,pool.get(i,'target'))
        def group(rr):
            return {'n':len(rr),'rms_uint8_mean':float(np.mean([r['rms_uint8'] for r in rr])) if rr else None,
                    'zero_pixel_fraction_in_support':float(np.mean([r['unchanged_pixel_fraction_in_support'] for r in rr if r['unchanged_pixel_fraction_in_support'] is not None])) if any(r['unchanged_pixel_fraction_in_support'] is not None for r in rr) else None}
        result[partition]={'n_composites':len(rows),'actual_embedding_calls':sum(r['embed_calls'] for r in rows),'unique_donor_sources':len({r['source_id'] for r in rows}),'unique_all_sources':len(ids[partition]),
            'bucket_counts':{str(k):len(v) for k,v in enumerate(pool.buckets)},'operation_counts':{op:sum(r['operation']==op for r in rows) for op in sorted({r['operation'] for r in rows})},
            'fg_le1':group([r for r in rows if 0<r['foreground_fraction']<=.01]),'fg_1to5':group([r for r in rows if .01<r['foreground_fraction']<=.05]),
            'clean_le1':group([r for r in rows if 0<1-r['foreground_fraction']<=.01]),'clean_1to5':group([r for r in rows if .01<1-r['foreground_fraction']<=.05]),
            'checked_twins_and_majority_targets':100,'manifests':{str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob('*/manifest.jsonl')}}
    assert not ids['train']&ids['dev'];result['source_overlap']=0
    atomic_json(a.root/f'triforensic-data/{a.family}/audit.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
