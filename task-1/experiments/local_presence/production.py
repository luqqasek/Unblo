"""Frozen full-data continuation and test inference; no test-label access."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .data_finetune import Replay
from .scan_empty import infer,scan_boxes
from experiments.native_localizer.common import seed_all,atomic_json,rgb_tensor,sha256,rng_state
from experiments.native_localizer.losses import segmentation
from experiments.message_localizer.model import MessageLocalizer
from watermark_localization.submission import save_submission,validate_submission
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

def pixelseal(root):
    out=root/'local-presence-production/pixelseal';out.mkdir(parents=True,exist_ok=True);seed_all(4242);started=time.monotonic();source=root/'small-zoom-production/pixelseal/last.pt'
    cfg={'family':'pixelseal','source':str(source),'source_sha256':sha256(source),'steps':150,'seed':4242,'batch':16,'synthetic':12,'real':4,'encoder_lr':2e-6,'new_lr':2e-5,'threshold':.5,'source_policy':'exact128fixed replay as ps-data-small validation','train_real_count':170,'test_recipe_frozen':True,'tta':False}
    atomic_json(out/'config.json',cfg);model=MessageLocalizer('pixelseal').cuda()
    if not (out/'last.pt').exists():
        state=torch.load(source,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);del state;model.freeze_encoder(False);model.train();replay=Replay(root,'pixelseal',False);ds=LocalisationDataset('data','train',methods=['pixelseal']);real=[ds[i] for i in range(len(ds))];assert len(real)==170
        enc=list(model.encoder.net.parameters());ids={id(p) for p in enc};new=[p for p in model.parameters() if id(p) not in ids and p.requires_grad];opt=torch.optim.AdamW([{'params':enc,'lr':2e-6},{'params':new,'lr':2e-5}],weight_decay=.01);rng=np.random.default_rng(4242);rrng=np.random.default_rng(12342)
        for step in range(150):
            examples=[replay.sample(rng) for _ in range(12)];items=[real[int(i)] for i in rrng.choice(len(real),4,replace=True)];x=torch.stack([rgb_tensor(v[0]) for v in examples]+[rgb_tensor(it['image']) for it in items]).cuda();y=torch.stack([torch.tensor(v[1]).float() for v in examples]+[torch.tensor(it['mask128']).float() for it in items]).cuda();bits=torch.tensor([list(v[2]) for v in examples]+[it['message'].tolist() for it in items],device='cuda').float()
            opt.zero_grad(set_to_none=True);loss=segmentation(model(x,bits)['logits'],y);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
            if (step+1)%50==0:
                print('PS_FULL_REFIT',step+1,'seconds',time.monotonic()-started,flush=True);torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step+1,'config':cfg,'rng':rng_state(),'data_rng':rng.bit_generator.state,'real_rng':rrng.bit_generator.state},out/'resume.pt')
        torch.save({'model':model.state_dict(),'config':cfg,'step':150},out/'last.pt');del real,replay,opt,x,y,bits;torch.cuda.empty_cache()
    else:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert state['config']==cfg;model.load_state_dict(state['model']);del state
    model.eval();test=LocalisationDataset('data','test',methods=['pixelseal']);assert len(test)==340;probs=[];names=[]
    with torch.inference_mode():
        for begin in range(0,len(test),16):
            items=[test[i] for i in range(begin,min(begin+16,len(test)))];x=torch.stack([rgb_tensor(it['image']) for it in items]).cuda();b=torch.tensor(np.array([it['message'] for it in items]),device='cuda').float();prob=model(x,b)['logits'].sigmoid().cpu().numpy();assert np.isfinite(prob).all();probs.extend(prob);names.extend(it['image_name'] for it in items)
    masks=torch.tensor(np.array(probs)>=.5,dtype=torch.uint8);dest=out/'pixelseal-subset.pt';save_submission(dest,names,masks,names);validate_submission(torch.load(dest,weights_only=True),names);np.savez_compressed(out/'test_probabilities.npz',names=names,probabilities=probs)
    atomic_json(out/'inference.json',{'n':340,'seconds':time.monotonic()-started,'checkpoint_sha256':sha256(out/'last.pt'),'subset_sha256':sha256(dest),'test_labels_used':False,'threshold':.5});print('PS_PRODUCTION_DONE',time.monotonic()-started,flush=True)
    del model,x,b;torch.cuda.empty_cache()

def scan(root,part):
    out=root/f'local-presence-production/tm-scan-{part}';out.mkdir(parents=True,exist_ok=True);dest=out/'proposals.json';started=time.monotonic();base=torch.load(root/'small-zoom-production/trustmark/trustmark-subset.pt',map_location='cpu',weights_only=True);lookup=dict(zip(base['image_names'],base['masks']));ds=LocalisationDataset('data','test',methods=['trustmark']);eligible=[i for i,r in enumerate(ds.meta) if not lookup[r['image_name']].any()];assert len(eligible)==38
    # GPU0 first refits PS, then handles8 cases; GPUs1..3 handle10 each.
    selected=eligible[:8] if part==0 else eligible[8+(part-1)*10:8+part*10];atomic_json(out/'config.json',{'part':part,'eligibility':'EMPTY baseline #2409 masks, no GT','selected':[ds.meta[i]['image_name'] for i in selected],'all_eligible':[ds.meta[i]['image_name'] for i in eligible],'acceptance':.9,'scan':'same frozen46,642box grid and4/2/1 exact refinement as validation','recipe_frozen_before_test':True})
    wm=load_watermark('trustmark',models_dir='checkpoints',device='cuda');boxes=scan_boxes();rows=json.loads(dest.read_text()) if dest.exists() else [];seen={r['image_name'] for r in rows}
    for j,i in enumerate(selected):
        it=ds[i]
        if it['image_name'] in seen:continue
        r=infer(it['image'],it['message'],it['image_name'],wm,boxes);rows.append(r);atomic_json(dest,rows);print('TM_TEST_SCAN',part,j+1,len(selected),'seconds',time.monotonic()-started,'bit_accuracy',r['correct_top'][0]['bit_accuracy'],flush=True)
    atomic_json(out/'completed.json',{'n':len(rows),'seconds':time.monotonic()-started,'accepted':sum(r['correct_top'][0]['bit_accuracy']>=.9 for r in rows),'wrong_accepts':sum(r['wrong_max_accuracy']>=.9 for r in rows),'test_labels_used':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--part',type=int,choices=range(4),required=True);a=p.parse_args();torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.part==0:pixelseal(a.root)
    scan(a.root,a.part)
if __name__=='__main__':main()
