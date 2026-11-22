"""Follow-up: RF83 correction, ordinary BCE plus either both-class Dice or IoU."""
import argparse,json,time,fcntl
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import Presence,soft_iou
from .data import full_examples
from .train import summary
from experiments.native_localizer.common import atomic_json,rgb_tensor,seed_all
from experiments.native_localizer.losses import segmentation
from experiments.native_localizer.evaluate import image_metrics
from experiments.message_localizer.model import MessageLocalizer
from wmloc.data import LocalisationDataset
from experiments.small_zoom.proposals import boxmask

class Spatial(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(65,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,3,padding=2,dilation=2),nn.GELU(),nn.Conv2d(32,1,3,padding=4,dilation=4))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,z,f):return z+8*torch.tanh(self.net(torch.cat([f,z[:,None].clamp(-8,8)/8],1)))[:,0]

def features(root,source='pixelseal-presence',family='pixelseal'):
    path=root/f'local-presence-runs/{source}/spatial-cache.pt'
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
        model=Presence().cuda();state=torch.load(path.parent/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['local']);del state;model.eval().requires_grad_(False)
        recipe='control-adapt' if family=='pixelseal' else 'objects-adapt'
        base=MessageLocalizer(family).cuda();state=torch.load(root/f'region-runs/{family}/{recipe}/fold1/model.pt',map_location='cpu',weights_only=False);base.load_state_dict(state['model']);del state;base.eval().requires_grad_(False)
        ds=LocalisationDataset('data','train',methods=[family]);data={}
        for name,examples in [('real',((ds[i]['image'],ds[i]['mask128'],ds[i]['message'],ds[i]['image_name']) for i in range(len(ds)))),('synthetic',((x,y,b,str(i)) for i,(x,y,b) in enumerate(full_examples(root,family,False))))]:
            fs=[];zs=[];ys=[];names=[]
            with torch.no_grad():
                for image,y,b,n in examples:
                    x=rgb_tensor(image)[None].cuda();bits=torch.tensor(b)[None].cuda().float();_,f=model(x);z=base(x,bits)['logits'];fs.append(f[0].float().cpu());zs.append(z[0].cpu());ys.append(torch.tensor(y).float());names.append(n)
            data[name]={'features':torch.stack(fs),'logits':torch.stack(zs),'targets':torch.stack(ys),'names':names}
        torch.save(data,path);del model,base;torch.cuda.empty_cache();return data

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--loss',choices=['dice','iou'],required=True);p.add_argument('--source',default='pixelseal-presence');p.add_argument('--run-name');p.add_argument('--family',choices=['pixelseal','trustmark'],default='pixelseal');a=p.parse_args()
    torch.set_num_threads(6);seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'local-presence-runs'/(a.run_name or f'ps-spatial-{a.loss}');out.mkdir(parents=True,exist_ok=True);assert not (out/'completed.json').exists();start=time.monotonic()
    d=features(a.root,a.source,a.family);names=d['real']['names'];split=json.loads(Path('experiments/native_localizer/reconstructed_splits.json').read_text())['families'][a.family][0];valid=[i for i,n in enumerate(names) if n in split['validation']];train=[i for i in range(len(names)) if i not in valid]
    rectangles={r['image_name']:r for r in json.loads((a.root/'small-zoom-finalscale/trustmark/proposals.json').read_text())} if a.family=='trustmark' else {}
    for group in d.values():
        for key in ['features','logits','targets']:group[key]=group[key].cuda()
    m=Spatial().cuda();opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=.01);rng=np.random.default_rng(42)
    atomic_json(out/'config.json',{'loss':a.loss,'steps':600,'batch':16,'real':4,'synthetic':12,'lr':2e-4,'seed':42,'fold':1,'effective_rf':83,'base_and_local_frozen':True,'threshold':.5,'endpoint_fixed':600,'precision':'FP32','followup':'reduce class-imbalance overcorrection and add spatial aggregation; developed after negative first series'})
    def evaluate(step):
        rows=[];probs=[];gt=[];ns=[]
        with torch.no_grad():
            for i in valid:
                g=d['real'];prob=m(g['logits'][i:i+1],g['features'][i:i+1].float())[0].sigmoid().cpu().numpy();y=g['targets'][i].cpu().numpy()
                proposal=rectangles.get(names[i],{}).get('correct_top',[])
                if proposal and proposal[0]['bit_accuracy']>=.9:prob[boxmask(proposal[0]['box'])]=1
                r=image_metrics(prob,y);r['image_name']=names[i];rows.append(r);probs.append(prob);gt.append(y);ns.append(names[i])
        atomic_json(out/f'correction-{step}.json',{'summary':summary(rows),'per_image':rows});np.savez_compressed(out/f'maps-{step}.npz',names=ns,probabilities=probs,targets=gt)
        print('SPATIAL_VALIDATION',a.loss,step,json.dumps(summary(rows)),flush=True);return rows
    baseline=evaluate(0)
    for step in range(600):
        idx=rng.choice(train,4,replace=True);si=rng.integers(len(d['synthetic']['names']),size=12)
        z=torch.cat([d['real']['logits'][idx],d['synthetic']['logits'][si]]);f=torch.cat([d['real']['features'][idx],d['synthetic']['features'][si]]).float();y=torch.cat([d['real']['targets'][idx],d['synthetic']['targets'][si]])
        zz=m(z,f);loss=segmentation(zz,y) if a.loss=='dice' else .5*F.binary_cross_entropy_with_logits(zz,y)+.5*soft_iou(zz,y)
        loss=loss+.002*(zz-z).square().mean();opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(norm);opt.step()
        if (step+1)%100==0:print('SPATIAL_TRAIN',a.loss,step+1,float(loss),time.monotonic()-start,flush=True)
        if step+1 in [200,600]:final=evaluate(step+1)
    torch.save({'model':m.state_dict(),'config':json.loads((out/'config.json').read_text())},out/'last.pt')
    atomic_json(out/'completed.json',{'baseline':summary(baseline),'final':summary(final),'delta_miou':float(np.mean([r['miou'] for r in final])-np.mean([r['miou'] for r in baseline])),'seconds':time.monotonic()-start,'peak_bytes':torch.cuda.max_memory_allocated()})
    print('SPATIAL_DONE',a.loss,(out/'completed.json').read_text(),flush=True)
if __name__=='__main__':main()
