"""Matched 200-update PixelSeal screen: control, tiny crop data, residual auxiliary."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.message_localizer.model import MessageLocalizer
from experiments.message_localizer.train import evaluate
from experiments.message_localizer.adapt import eval_indices
from experiments.native_localizer.common import atomic_json,append_jsonl,seed_all,rng_state,rgb_tensor
from experiments.native_localizer.losses import segmentation,pair_loss
from experiments.native_localizer.evaluate import image_metrics
from wmloc.data import LocalisationDataset
from .tiny_data import TinyData

class ResidualModel(MessageLocalizer):
    def __init__(self):
        super().__init__('pixelseal');self.residual=nn.Conv2d(32,16,1)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        self.context.register_forward_hook(self.capture)
    def capture(self,module,args,output):self.local_features=output
    def forward(self,x,message):
        result=super().forward(x,message)
        result['residual_y']=F.pixel_shuffle(self.residual(self.local_features),4)[:,0]
        return result

def aux_loss(pred,target,regions,valid):
    if not valid.any():return pred.sum()*0
    error=(pred[valid].float()-target[valid].float()).abs();mask=regions[valid]
    return .5*((error*mask).sum((-2,-1))/mask.sum((-2,-1)).clamp_min(1)+(error*~mask).sum((-2,-1))/(~mask).sum((-2,-1)).clamp_min(1)).mean()

def build(state):
    seed_all(941);m=ResidualModel().cuda();missing,unexpected=m.load_state_dict(state,strict=False)
    assert not unexpected and set(missing)<= {'residual.weight','residual.bias'}
    m.freeze_encoder(False);m.train();return m

def optimizer(m,lr1,lr2):
    enc=list(m.encoder.net.parameters());ids={id(p) for p in enc};new=[p for p in m.parameters() if id(p) not in ids and p.requires_grad]
    return torch.optim.AdamW([{'params':enc,'lr':lr1},{'params':new,'lr':lr2}],weight_decay=.01)

def backward_micro(model,x,y,b,pairs,d,region,valid,useaux,aux_weight,pair_weight):
    """Exact per-image/pair reductions with micro16 and effective batch32."""
    totals=[0.,0.,0.];n=len(x);nv=int(valid.sum())
    for start in range(0,n,16):
        end=min(n,start+16);pp=[(a-start,c-start) for a,c in pairs if start<=a<end and start<=c<end]
        assert not any((start<=a<end)!=(start<=c<end) for a,c in pairs)
        output=model(x[start:end],b[start:end]);seg=segmentation(output['logits'],y[start:end]);pair=pair_loss(output['logits'],y[start:end],pp)
        stop=min(end,len(d));local_valid=valid[start:stop]
        aux=aux_loss(output['residual_y'][:stop-start],d[start:stop].cuda(),region[start:stop].cuda(),local_valid.cuda()) if useaux and stop>start else seg*0
        ws=(end-start)/n;wp=len(pp)/max(1,len(pairs));wa=int(local_valid.sum())/max(1,nv)
        loss=ws*seg+pair_weight*wp*pair+aux_weight*wa*aux;loss.backward()
        for j,value in enumerate([ws*seg,wp*pair,wa*aux]):totals[j]+=float(value.detach())
    return totals

@torch.inference_mode()
def tiny_eval(model,dev,out):
    model.eval();rng=np.random.default_rng(2703);rows=[];res=[]
    for _ in range(8):
        x,y,b,pairs,d,region,valid=dev.batch(rng,16,True);z=model(x.cuda(),b.cuda());prob=z['logits'].sigmoid().cpu().numpy();rp=z['residual_y'].cpu()
        for i in range(len(x)):
            if valid[i]:
                row=image_metrics(prob[i],y[i].numpy());row['image_name']=str(len(rows));rows.append(row)
                nz=d[i]!=0
                if nz.any():res.append({'mae_nonzero':float((rp[i][nz]-d[i][nz]).abs().mean()),'zero_predictor_mae_nonzero':float(d[i][nz].abs().mean())})
    atomic_json(out/'tiny_dev.json',{'rows':rows,'residual':res});model.train()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=['control','tiny','tiny_aux'],required=True);p.add_argument('--adapt',action='store_true');p.add_argument('--folds',nargs='+',type=int,default=[1,2,3]);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'tiny-runs'/a.variant;out.mkdir(parents=True,exist_ok=True);tiny=a.variant!='control';useaux=a.variant=='tiny_aux'
    data=TinyData(a.root);start=time.monotonic();counts=Counter()
    cfg={'variant':a.variant,'pretrain_updates':200,'batch':32,'seed':941,'init':'region-runs/pixelseal/control/last.pt','encoder_lr':5e-6,'new_lr':5e-5,'aux_weight':.1 if useaux else 0,'aux':'FP32 Y residual,16 subpixels per128cell, balanced L1 within/outside rectangle; no residual inference input','threshold':.5}
    atomic_json(out/'config.json',cfg)
    if not a.adapt:
        assert not (out/'last.pt').exists(),'Preserve completed experiment'
        state=torch.load(a.root/'region-runs/pixelseal/control/last.pt',map_location='cpu',weights_only=False)['model'];model=build(state);del state
        opt=optimizer(model,5e-6,5e-5);rng=np.random.default_rng(941)
        for step in range(200):
            x,y,b,pairs,d,region,valid=data.batch(rng,32,tiny);x=x.cuda();y=y.cuda();b=b.cuda()
            opt.zero_grad(set_to_none=True);seg,pair,aux=backward_micro(model,x,y,b,pairs,d,region,valid,useaux,.1*min(1,(step+1)/50),.1)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
            areas=y.mean((-2,-1)).detach().cpu().numpy();counts['views']+=len(areas);counts['fg_le1']+=int(((areas>0)&(areas<=.01)).sum());counts['fg_1to5']+=int(((areas>.01)&(areas<=.05)).sum());counts['empty']+=int((areas==0).sum())
            if (step+1)%25==0:
                row={'variant':a.variant,'step':step+1,'seg':seg,'aux':aux,'seconds':time.monotonic()-start,'peak_bytes':torch.cuda.max_memory_allocated()};append_jsonl(out/'train.jsonl',row);print(json.dumps(row),flush=True)
            if (step+1)%100==0:torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step+1,'config':cfg,'rng':rng_state(),'data_rng':rng.bit_generator.state},out/'resume.pt')
        torch.save({'model':model.state_dict(),'config':cfg,'step':200},out/'last.pt');atomic_json(out/'exposure.json',dict(counts))
        dev=TinyData(a.root,'dev',False);tiny_eval(model,dev,out)
        real=evaluate(model,'pixelseal',a.root,out/'real');atomic_json(out/'completed.json',{'seconds':time.monotonic()-start,'real_miou':real});print('SYNTHETIC_ONLY_REAL',a.variant,real,flush=True)
    else:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)['model']
        ds=LocalisationDataset('data','train',methods=['pixelseal']);lookup={r['image_name']:i for i,r in enumerate(ds.meta)}
        splits=json.loads(Path('experiments/native_localizer/reconstructed_splits.json').read_text())['families']['pixelseal'];allrows=[]
        for fold,split in enumerate(splits,1):
            if fold not in a.folds:continue
            dest=out/'adapt'/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'per_image.json').exists():allrows+=json.loads((dest/'per_image.json').read_text());continue
            valid_idx=[lookup[n] for n in split['validation']];train=[i for i in range(len(ds)) if i not in set(valid_idx)];assert not set(train)&set(valid_idx)
            atomic_json(dest/'split.json',{'train':[ds.meta[i]['image_name'] for i in train],'validation':split['validation']})
            model=build(state);opt=optimizer(model,2e-6,2e-5);rng=np.random.default_rng(941);real_rng=np.random.default_rng(12342)
            for step in range(150):
                x,y,b,pairs,d,region,valid=data.batch(rng,24,tiny);items=[ds[int(i)] for i in real_rng.choice(train,8,replace=True)]
                x=torch.cat([x,torch.stack([rgb_tensor(it['image']) for it in items])]).cuda();y=torch.cat([y,torch.stack([torch.tensor(it['mask128']).float() for it in items])]).cuda();b=torch.cat([b,torch.tensor(np.array([it['message'] for it in items])).float()]).cuda()
                opt.zero_grad(set_to_none=True);backward_micro(model,x,y,b,pairs,d,region,valid,useaux,.1,.05)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
                if (step+1)%50==0:print('ADAPT',a.variant,fold,step+1,'seconds',time.monotonic()-start,flush=True)
            torch.save({'model':model.state_dict(),'config':cfg,'fold':fold},dest/'model.pt');rows=eval_indices(model,ds,valid_idx,dest);allrows+=rows
            print('OUTER',a.variant,fold,float(np.mean([r['miou'] for r in rows])),flush=True);del model,opt;torch.cuda.empty_cache()
        assert len(allrows)==len({r['image_name'] for r in allrows});atomic_json(out/('adapt/per_image.json' if len(a.folds)==3 else 'adapt/screen_per_image.json'),allrows);print('OOF' if len(a.folds)==3 else 'SINGLE_FOLD_SCREEN',a.variant,len(allrows),float(np.mean([r['miou'] for r in allrows])),flush=True)
if __name__=='__main__':main()
