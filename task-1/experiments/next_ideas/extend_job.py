"""One bounded continuation; fixed450 endpoint, never selected on public scores."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from .audit import fallback
from experiments.native_localizer.common import atomic_json,sha256
from experiments.native_localizer.evaluate import image_metrics
from experiments.triforensic_wm.evaluate import summarize


def main():
    root=Path('/p/project1/training2666/pszenny1/native-localizer');old=root/'next-ideas-runs';new=root/'next-ideas-extended-runs';new.mkdir(exist_ok=True)
    protocol=json.loads((old/'extension_protocol.json').read_text());assert protocol['endpoint']==450
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==4
    logs=new/'logs';logs.mkdir(exist_ok=True);workers=[];handles=[];start=time.monotonic()
    for gpu in range(4):
        name=f'fold{gpu+1}' if gpu<3 else 'all170';run=new/'stegastamp'/name
        if (run/'completed.json').exists():continue
        args=['--fold',str(gpu+1)] if gpu<3 else ['--all-real']
        if (run/'last.pt').exists():args+=['--resume']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
        log=(logs/f'{name}.log').open('a');handles.append(log)
        workers.append(subprocess.Popen([sys.executable,'-u','-m','experiments.next_ideas.stegastamp_adapt','--root',str(root),'--continue150','--steps','450','--wall-seconds','850',*args],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in workers]
    for h in handles:h.close()
    assert not any(codes),codes
    for name in ('fold1','fold2','fold3','all170'):
        assert json.loads((new/'stegastamp'/name/'completed.json').read_text())['step']==450
    before=[];after=[];raw=[]
    cfg=json.loads((old/'selected_recipe.json').read_text())
    for fold in (1,2,3):
        z0=np.load(old/f'stegastamp/fold{fold}/maps-150.npz');z=np.load(new/f'stegastamp/fold{fold}/maps-450.npz');assert np.array_equal(z0['names'],z['names']) and np.array_equal(z0['targets'],z['targets'])
        for n,p0,p,y in zip(z['names'],z0['probabilities'],z['probabilities'],z['targets']):
            a=image_metrics(p0,y);b=image_metrics(p,y);raw.append(dict(b,image_name=str(n),fold=fold,family='stegastamp'))
            if cfg['postprocess']=='top1pct':
                for dest,prob in ((a,p0),(b,p)):
                    scored=image_metrics(fallback(prob,'top1pct').astype(float),y)
                    for key in ('miou','fg_iou','clean_iou','empty','full','pred_area','complete_fg_miss','fp_area','internal_boundary_f1','minority_iou'):
                        if key in scored:dest[key]=scored[key]
            before.append(dict(a,image_name=str(n),fold=fold,family='stegastamp'));after.append(dict(b,image_name=str(n),fold=fold,family='stegastamp'))
    assert len(after)==len({r['image_name'] for r in after})==170
    delta=np.array([b['miou']-a['miou'] for a,b in zip(before,after)]);rng=np.random.default_rng(42);ci=np.quantile(delta[rng.integers(170,size=(5000,170))].mean(1),[.025,.975]);s0,s=summarize(before),summarize(after)
    small0=np.mean([r['fg_iou'] for r in before if r['true_area']<=.05]);small=np.mean([r['fg_iou'] for r in after if r['true_area']<=.05])
    guard=small>=small0-.01 and s['wm_over50']['clean_iou']>=s0['wm_over50']['clean_iou']-.01
    selected=delta.mean()>=protocol['minimum_gain'] and ci[0]>0 and guard
    result={'protocol':protocol,'before_summary':s0,'summary':s,'raw_summary':summarize(raw),'per_image':after,'raw_per_image':raw,'paired_delta':{'mean':float(delta.mean()),'ci95':ci.tolist(),'improved':int((delta>1e-8).sum()),'worse':int((delta<-1e-8).sum())},'minority_guard_passed':bool(guard),'selected':bool(selected),'seconds':time.monotonic()-start,'public_score_used':False}
    atomic_json(new/'evaluation.json',result);print('EXTENDED_450_EVALUATION',json.dumps({k:v for k,v in result.items() if k not in ('per_image','raw_per_image','summary','before_summary','raw_summary')}),flush=True)
    if not selected:return
    cfg.update(checkpoint=str(new/'stegastamp/all170/last.pt'),updates=450,validation_miou_before_fallback=summarize(raw)['all']['miou'],validation_miou_final=s['all']['miou'],continuation_selection=result['paired_delta'],evidence_sha256=sha256(new/'evaluation.json'),postprocess_reselected=False)
    atomic_json(new/'selected_recipe.json',cfg)
    out=root/'next-ideas-extended-production/stegastamp';out.mkdir(parents=True,exist_ok=True);workers=[];handles=[]
    for part in range(4):
        if (out/f'part{part}.json').exists():continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[part],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6');log=(out/f'worker{part}.log').open('a');handles.append(log)
        workers.append(subprocess.Popen([sys.executable,'-u','-m','experiments.next_ideas.infer','--root',str(root),'--part',str(part),'--extended'],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in workers]
    for h in handles:h.close()
    assert not any(codes),codes;print('EXTENDED_PRODUCTION_READY',time.monotonic()-start,flush=True)


if __name__=='__main__':main()
