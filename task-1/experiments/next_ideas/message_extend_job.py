"""Four existing parent runs continued on one node; no new test inference."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments.native_localizer.common import atomic_json

def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer');out=root/'message-extended-runs';out.mkdir(exist_ok=True)
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==4
    protocol={'endpoint':450,'parent_updates':150,'new_updates':300,'fold':1,'threshold':.5,'effective_batch':16,'real':4,'synthetic':12,'minimum_gain':.005,'minority_guard':.01,'arms':['ps-data-small','tm-data-small','ps-data-online','tm-data-online'],'public_score_used':False,'selection':'Exploratory fixedfold, compare safe capped verification at both endpoints; also report original0. No test inference without evidence review.'}
    atomic_json(out/'protocol.json',protocol);workers=[];handles=[];start=time.monotonic()
    for gpu,(family,online) in enumerate([('pixelseal',False),('trustmark',False),('pixelseal',True),('trustmark',True)]):
        name=('ps' if family=='pixelseal' else 'tm')+'-data-'+('online' if online else 'small')
        if (out/name/'completed.json').exists():continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
        h=(out/(name+'.log')).open('a');handles.append(h)
        workers.append(subprocess.Popen([sys.executable,'-u','-m','experiments.next_ideas.message_extend','--root',str(root),'--family',family,*(['--online'] if online else [])],env=env,stdout=h,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in workers]
    for h in handles:h.close()
    assert not any(codes),codes
    results={}
    for name in protocol['arms']:
        parent=root/'local-presence-runs'/name;run=out/name
        a=json.loads((run/'correction-150.json').read_text());b=json.loads((run/'correction-450.json').read_text());old=json.loads((parent/'correction-0.json').read_text())
        before={r['image_name']:r for r in a['safe_per_image']};rows=b['safe_per_image'];delta=np.array([r['miou']-before[r['image_name']]['miou'] for r in rows]);rng=np.random.default_rng(42)
        ci=np.quantile(delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1),[.025,.975]);s0,s=a['safe_summary'],b['safe_summary']
        tiny=lambda rr:float(np.mean([r['fg_iou'] for r in rr if 0<r['true_area']<=.05]))
        guard=tiny(rows)>=tiny(a['safe_per_image'])-.01 and s['wm_over50']['clean_iou']>=s0['wm_over50']['clean_iou']-.01
        results[name]={'before':s0,'after':s,'original0_historical':old['summary'],'delta':float(delta.mean()),'ci95':ci.tolist(),'improved':int((delta>0).sum()),'worse':int((delta<0).sum()),'minority_guard':bool(guard),'candidate':bool(delta.mean()>=.005 and ci[0]>0 and guard),'legacy450':b['summary'],'per_image':rows}
    atomic_json(out/'evaluation.json',{'protocol':protocol,'results':results,'seconds':time.monotonic()-start})
    print('MESSAGE_EXTENDED_DONE',json.dumps({k:{kk:v[kk] for kk in ('delta','ci95','candidate')} for k,v in results.items()}),flush=True)
if __name__=='__main__':main()
