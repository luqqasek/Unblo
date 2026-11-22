"""Single-loop shared full-backbone clean-hole screening; frozen GT-free routing."""
import argparse,json,time,math,importlib.metadata,gc
from pathlib import Path
import numpy as np
import torch
from .model import IMLHoles,PIN,WEIGHT_HASH
from experiments.native_localizer.common import seed_all,rgb_tensor,atomic_json,append_jsonl,rng_state,sha256
from experiments.native_localizer.generator import EncodedDataset,minority_mask
from experiments.local_presence.data import down
from experiments.native_localizer.losses import segmentation
from experiments.native_localizer.evaluate import image_metrics
from experiments.triforensic_wm.evaluate import summarize
from experiments.next_ideas.audit import fallback
from experiments.small_zoom.proposals import boxmask
from experiments.local_presence.verification import verified
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

FAMILIES=('pixelseal','trustmark','stegastamp')
BINARY_KEYS=('miou','fg_iou','clean_iou','empty','full','pred_area','complete_fg_miss','fp_area','internal_boundary_f1','minority_iou')

def baselines(root):
    paths={'pixelseal':root/'local-presence-runs/ps-data-small/maps-150.npz','trustmark':root/'region-runs/trustmark/objects-adapt/fold1/maps.npz','stegastamp':root/'next-ideas-extended-runs/stegastamp/fold1/maps-450.npz'}
    result={};hashes={}
    for family,path in paths.items():
        z=np.load(path);hashes[family]=sha256(path)
        for name,p,y in zip(z['names'],z['probabilities'],z['targets']):result[str(name)]={'p':p.squeeze().copy(),'y':y.squeeze().astype(bool),'family':family}
    for path in [root/'small-zoom-finalscale/trustmark/proposals.json',*[root/f'local-presence-runs/tm-scan-{k}/proposals.json' for k in (0,1)]]:
        hashes[str(path)]=sha256(path)
        for r in json.loads(path.read_text()):
            if r['image_name'] in result and r.get('correct_top') and verified(r['correct_top'][0]):
                box=r['correct_top'][0]['box'];area=(box[2]-box[0])*(box[3]-box[1])/512**2
                assert area<=.15,('Large historical override forbidden',r['image_name'],area)
                result[r['image_name']]['p'][boxmask(box)]=1.
    for n,r in result.items():r['mask']=fallback(r['p'],'top1pct') if r['family']=='stegastamp' else r['p']>=.5
    assert len(result)==171
    return result,hashes

def score(p,mask,y,**meta):
    r=image_metrics(p,y);binary=image_metrics(mask.astype(np.float32),y)
    for k in BINARY_KEYS:
        if k in binary:r[k]=binary[k]
    return dict(r,**meta)

def synth_bank(root):
    rng=np.random.default_rng(20261122);examples=[];provenance=[]
    for family in FAMILIES:
        ds=EncodedDataset(root/f'encoded/{family}/pilot-train/encoded.jsonl',family);full=[i for i,r in enumerate(ds.rows) if r['rectangle']==[0,0,512,512]]
        for j in range(128):
            i=int(rng.choice(full));clean,marked,row=ds.load(i);fraction=float(np.exp(rng.uniform(np.log(.003),np.log(.3))));shape=('rectangle','strip','components')[j%3];hole=minority_mask(rng,fraction,shape)
            donor=row;replacement=clean
            if j%2:
                k=int(rng.choice(full));replacement,_,donor=ds.load(k)
            if j%16==0:hole[:]=False # Intact hard negatives, including naturally low-signal areas.
            x=np.where(hole[...,None],replacement,marked).astype(np.uint8);y=1-down(~hole)
            examples.append((rgb_tensor(x),torch.tensor(y).float()))
            provenance.append({'family':family,'source_id':row['source_id'],'donor_source':donor['source_id'],'source_partition':row['partition'],'donor_partition':donor['partition'],'realization_id':row['realization_id'],'initial_rectangle':row['rectangle'],'clean_fraction':float(y.mean()),'shape':shape,'same_content':j%2==0,'intact':j%16==0})
    assert all(r['source_partition']=='train' and r['donor_partition']=='train' for r in provenance)
    return examples,provenance

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--steps',type=int,default=300);a=p.parse_args()
    root=a.root;out=root/'iml-holes-runs/fold1';out.mkdir(parents=True,exist_ok=True);assert not (out/'completed.json').exists()
    seed_all(42);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.monotonic()
    ds=LocalisationDataset('data','train');items=[ds[i] for i in range(len(ds))];splits=json.loads(Path('experiments/native_localizer/reconstructed_splits.json').read_text())['families'];validnames={n for f in FAMILIES for n in splits[f][0]['validation']};valid=[i for i,r in enumerate(items) if r['image_name'] in validnames];train=[i for i,r in enumerate(items) if r['image_name'] not in validnames and r['full_watermark']];assert len(valid)==171
    base,hashes=baselines(root);assert set(base)==validnames
    for i in valid:assert np.array_equal(base[items[i]['image_name']]['y'],items[i]['mask128'])
    # Recompute for EVERY validation image. Prior oracle files were GT-filtered;
    # missing rows must never be interpreted as a negative routing decision.
    agreement={};decoder_rows=[]
    for family in FAMILIES:
        wm=load_watermark(family,models_dir='checkpoints',device='cuda')
        family_names=set(splits[family][0]['validation']);ids=[i for i in valid if items[i]['image_name'] in family_names]
        with torch.inference_mode():
            for i in ids:
                it=items[i];z=wm.logits(rgb_tensor(it['image'])[None].cuda())[0].float().cpu().numpy();bits=np.asarray(it['message']);correct=int(((z>0)==bits).sum());accepted=correct>=math.ceil(.9*len(bits));agreement[it['image_name']]=accepted
                decoder_rows.append({'image_name':it['image_name'],'family':family,'correct_bits':correct,'nbits':len(bits),'accepted':accepted,'GT_used':False})
        del wm;gc.collect();torch.cuda.empty_cache()
    assert set(agreement)==validnames
    atomic_json(out/'full_decoder_all_validation.json',decoder_rows)
    atomic_json(out/'split.json',{'train':[items[i]['image_name'] for i in train],'validation':sorted(validnames),'all_fold_training':[r['image_name'] for r in items if r['image_name'] not in validnames]})
    images={i:rgb_tensor(r['image']) for i,r in enumerate(items)};targets={i:torch.tensor(1-r['mask128']).float() for i,r in enumerate(items)}
    model=IMLHoles(root).cuda();model.eval();assert torch.cuda.device_count()==4
    x=images[valid[0]][None].cuda()
    with torch.no_grad():
        t=time.monotonic();fp=model(x,force_fp32=True);torch.cuda.synchronize();fp_seconds=time.monotonic()-t
        t=time.monotonic();amp=model(x);torch.cuda.synchronize();amp_seconds=time.monotonic()-t
        t=time.monotonic();padded=model(x,padded=True);torch.cuda.synchronize();pad_seconds=time.monotonic()-t
    difference=(fp.sigmoid()-amp.sigmoid()).abs();probe={'fp32_seconds':fp_seconds,'bf16_seconds':amp_seconds,'padded1024_seconds':pad_seconds,'fp32_bf16_probability_max':float(difference.max()),'fp32_bf16_probability_mean':float(difference.mean()),'native_vs_padded_probability_mean_abs':float((amp.sigmoid()-padded.sigmoid()).abs().mean()),'raw_pixels_resampled':False,'input_projection':'FP32','parameters':sum(v.numel() for v in model.parameters()),'input512_shape':list(amp.shape),'padded_valid_output_shape':list(padded.shape),'source_commit':PIN,'weight_sha256':WEIGHT_HASH,'checkpoint_strict_loaded':True}
    atomic_json(out/'probe.json',probe);print('IML_PROBE',json.dumps(probe),flush=True)
    assert probe['fp32_bf16_probability_mean']<.01,'Investigate AMP before training'
    del fp,amp,padded,x;torch.cuda.empty_cache()
    forward=torch.nn.DataParallel(model,device_ids=list(range(4)));syn,provenance=synth_bank(root);atomic_json(out/'synthetic_provenance.json',provenance)
    groups={}
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        lr=1e-5 if name.startswith('net.encoder_net.') else 3e-5;wd=0. if param.ndim<=1 or name.endswith('.bias') or 'norm' in name else .01
        groups.setdefault((lr,wd),[]).append(param)
    opt=torch.optim.AdamW([{'params':v,'lr':k[0],'weight_decay':k[1]} for k,v in groups.items()]);rng=np.random.default_rng(42)
    config={'source_commit':PIN,'weight_sha256':WEIGHT_HASH,'steps':a.steps,'effective_batch':8,'real':4,'synthetic':4,'real_training_images':len(train),'validation_images':len(valid),'family_shared':True,'all_weights_trainable':True,'precision':'FP32 raw normalization and projection; BF16 deeper; FP32 loss','upstream_context_change':'native512, positional interpolation in upstream code; strict1024 parameter shapes','BN_running_stats':'fixed','stochastic_depth':0,'dropout':0,'threshold':.5,'router':'original full decoder>=90% AND baseline predicted watermark area>50%','fusion':'equal probability blend with inverted clean head on routed images; baseline elsewhere','baseline_hashes':hashes,'runtime_versions':{k:importlib.metadata.version(k) for k in ('torch','timm','fvcore','numpy','torchvision')},'new_embedding_calls':0,'loss':'existing per-image halfBCE halfdualDice','gpus':4}
    atomic_json(out/'config.json',config)
    @torch.inference_mode()
    def evaluate(step):
        model.eval();rows=[];brows=[];srows=[];names=[];pp=[];cp=[];routes=[];masks=[];ys=[]
        for begin in range(0,len(valid),8):
            ids=valid[begin:begin+8];xx=torch.stack([images[i] for i in ids]).cuda();clean=forward(xx).float().sigmoid().cpu().numpy()
            for i,c in zip(ids,clean):
                it=items[i];name=it['image_name'];b=base[name];y=it['mask128'];route=agreement[name] and b['mask'].mean()>.5;prob=.5*b['p']+.5*(1-c) if route else b['p'];mask=prob>=.5 if route else b['mask']
                meta={'image_name':name,'family':b['family'],'full_watermark':bool(it['full_watermark']),'routed':bool(route)}
                rows.append(score(prob,mask,y,**meta));brows.append(score(b['p'],b['mask'],y,**meta));srows.append(dict(image_metrics(1-c,y),**meta));names.append(name);pp.append(prob);cp.append(c);routes.append(route);masks.append(mask);ys.append(y)
        delta=np.array([r['miou']-b['miou'] for r,b in zip(rows,brows)]);boot=np.random.default_rng(42);ci=np.quantile(delta[boot.integers(len(delta),size=(5000,len(delta)))].mean(1),[.025,.975]);summ=summarize(rows);before=summarize(brows)
        family={f:{'before':summarize([r for r in brows if r['family']==f]),'after':summarize([r for r in rows if r['family']==f])} for f in FAMILIES}
        result={'step':step,'summary':summ,'baseline':before,'specialist_diagnostic':summarize(srows),'by_family':family,'paired_delta':{'mean':float(delta.mean()),'ci95':ci.tolist()},'routed':int(sum(routes)),'routed_non_restoration':sum(r['routed'] and not r['full_watermark'] for r in rows),'per_image':rows,'baseline_per_image':brows,'specialist_per_image':srows,'protocol':'predeclared equal blend, threshold0.5, original fold1 across3 families, reused development; no GT routing'}
        atomic_json(out/f'real-{step}.json',result);np.savez_compressed(out/f'maps-{step}.npz',names=names,probabilities=pp,clean_probabilities=cp,masks=masks,targets=ys,routed=routes)
        print('IML_VALIDATION',step,json.dumps({'miou':summ['all']['miou'],'baseline':before['all']['miou'],'delta':result['paired_delta'],'clean_wm_over50':summ['wm_over50']['clean_iou'],'families':{f:v['after']['all']['miou'] for f,v in family.items()},'routed':sum(routes)}),flush=True);model.train();return result
    evaluate(0);exposure=[];training=time.monotonic();stop='endpoint';final=None
    for step in range(1,a.steps+1):
        rid=rng.choice(train,4,replace=True);sid=rng.integers(len(syn),size=4);x=torch.stack([images[i] for i in rid]+[syn[i][0] for i in sid]).cuda();y=torch.stack([targets[i] for i in rid]+[syn[i][1] for i in sid]).cuda()
        model.train();opt.zero_grad(set_to_none=True);z=forward(x);loss=segmentation(z.float(),y.float());loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
        exposure.append({'step':step,'real':[items[i]['image_name'] for i in rid],'synthetic_indices':sid.tolist()})
        if step==1 or step%10==0:
            torch.cuda.synchronize();elapsed=time.monotonic()-training;row={'step':step,'loss':float(loss),'gradient_norm':float(norm),'elapsed':elapsed,'seconds_per_update_including_evaluations':elapsed/step,'remaining_training_estimate':elapsed/step*(a.steps-step),'peak_bytes':[torch.cuda.max_memory_allocated(i) for i in range(4)]};append_jsonl(out/'train.jsonl',row);print('IML_TRAIN',json.dumps(row),flush=True)
        if step in (50,150,a.steps):
            final=evaluate(step);torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':config,'rng':rng_state(),'data_rng':rng.bit_generator.state},out/'last.pt');atomic_json(out/'exposure.json',exposure)
            if step==150 and final['paired_delta']['mean']<=0 and final['summary']['wm_over50']['clean_iou']<=final['baseline']['wm_over50']['clean_iou']:stop='predeclared_no_gain_gate';break
        if time.monotonic()-start>1600:stop='wall_budget';final=evaluate(step);break
    assert final is not None
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':config,'rng':rng_state(),'data_rng':rng.bit_generator.state},out/'last.pt');atomic_json(out/'exposure.json',exposure)
    guard=all(v['after']['all']['miou']>=v['before']['all']['miou']-.01 for v in final['by_family'].values())
    for f in FAMILIES:
        arows=[r for r in final['per_image'] if r['family']==f and 0<r['true_area']<=.05];brows=[r for r in final['baseline_per_image'] if r['family']==f and 0<r['true_area']<=.05]
        guard &= np.mean([r['fg_iou'] for r in arows])>=np.mean([r['fg_iou'] for r in brows])-.01
    candidate=final['paired_delta']['mean']>=.005 and final['paired_delta']['ci95'][0]>0 and guard
    atomic_json(out/'completed.json',{'step':step,'stop':stop,'seconds':time.monotonic()-start,'candidate':bool(candidate),'minority_guard':bool(guard),'final_miou':final['summary']['all']['miou'],'delta':final['paired_delta'],'test_inference':False});print('IML_DONE',(out/'completed.json').read_text(),flush=True)
if __name__=='__main__':main()
