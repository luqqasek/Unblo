"""500-update matched continuation: synthetic only vs75/25 real patch adaptation."""
import argparse,json,time,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import PatchBank,down
from .model import Presence,balanced
from .train import local_eval
from experiments.native_localizer.common import seed_all,atomic_json
from wmloc.data import LocalisationDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--real',action='store_true');p.add_argument('--family',choices=['pixelseal','trustmark'],default='pixelseal');a=p.parse_args()
    torch.set_num_threads(6);seed_all(73);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    prefix='ps' if a.family=='pixelseal' else 'tm';name=prefix+('-adapt-real' if a.real else '-adapt-control');out=a.root/'local-presence-runs'/name;out.mkdir(parents=True,exist_ok=True);assert not (out/'last.pt').exists();start=time.monotonic()
    bank=PatchBank(a.root,a.family,count=512);ds=LocalisationDataset('data','train',methods=[a.family]);split=json.loads(Path('experiments/native_localizer/reconstructed_splits.json').read_text())['families'][a.family][0]
    valid=[i for i,r in enumerate(ds.meta) if r['image_name'] in split['validation']];train=[i for i in range(len(ds)) if i not in valid];rrng=np.random.default_rng(12342);rx=[];ry=[]
    for i in train:
        item=ds[i];image=item['image'];m=item['mask128'].astype(bool)
        # GT used only in outer TRAINING to center native64 patches; no resize.
        for polarity in [False,True]:
            points=np.argwhere(m==polarity)
            if not len(points):continue
            for j in range(4):
                cy,cx=points[int(rrng.integers(len(points)))];x=int(np.clip(cx*4-32,0,448));y=int(np.clip(cy*4-32,0,448))
                rx.append(image[y:y+64,x:x+64].copy());ry.append(m[y//4:y//4+16,x//4:x//4+16].astype(np.float32))
    rx=torch.tensor(np.stack(rx)).permute(0,3,1,2).cuda();ry=torch.tensor(np.stack(ry)).cuda()
    model=Presence().cuda();state=torch.load(a.root/f'local-presence-runs/{a.family}-presence/last.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['local']);del state
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);rng=np.random.default_rng(73)
    atomic_json(out/'config.json',{'family':a.family,'real':a.real,'updates':500,'batch':128,'real_views':32 if a.real else 0,'synthetic_views':96 if a.real else 128,'lr':1e-4,'fold':1,'init':a.family+'-presence1000','real_patch_names':[ds.meta[i]['image_name'] for i in train],'local_rf':27,'seed':73})
    for step in range(500):
        x,y=bank.batch(rng,48 if a.real else 64);ns=len(x)
        if a.real:
            ids=rrng.integers(len(rx),size=32);x=torch.cat([x,rx[ids].float()/255]);y=torch.cat([y,ry[ids]])
        z,_=model(x);changed=y[:ns:2]!=y[1:ns:2];direction=torch.sign(y[1:ns:2]-y[:ns:2]);rank=F.softplus(1-direction*(z[1:ns:2]-z[:ns:2]));pair=rank[changed].mean() if changed.any() else z.sum()*0
        patch=F.binary_cross_entropy_with_logits(torch.logsumexp(z.flatten(1),1)-np.log(256),y.flatten(1).amax(1));loss=.5*balanced(z,y)+.5*F.binary_cross_entropy_with_logits(z,y)+.1*pair+.05*patch
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
        if (step+1)%100==0:print('ADAPT_BRANCH',name,step+1,float(loss),time.monotonic()-start,flush=True)
        if (step+1)%250==0:torch.save({'local':model.state_dict(),'optimizer':opt.state_dict(),'step':step+1,'rng':rng.bit_generator.state},out/'resume.pt')
    metrics=local_eval(model,ds,valid,out/'local-real-1500.json');torch.save({'local':model.state_dict(),'step':1500},out/'last.pt');atomic_json(out/'pretrain_completed.json',{'seconds':time.monotonic()-start,'summary':metrics})
    del model,opt,rx,ry;torch.cuda.empty_cache()
    subprocess.run([sys.executable,'-u','-m','experiments.local_presence.spatial','--root',str(a.root),'--loss','iou','--source',name,'--family',a.family,'--run-name',name+'-spatial'],check=True)
if __name__=='__main__':main()
