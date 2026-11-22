"""Post-training diagnostics only: never select a head/checkpoint on these measurements."""
import os,json,argparse
import numpy as np
import torch
from torch.nn import functional as F
from .prepare import ROOT,OUT,FAMILIES,save
from .train import Data,metrics,summarize

@torch.inference_mode()
def collect(model,data,ids,synthetic=False):
    model.eval();rows=[]
    for i in ids:
        b=data.batch([i]);o=model(b);y=b['mask128'].cpu().numpy()
        # Official majority rule applied to the binary auxiliary 512 prediction.
        aux=(F.avg_pool2d((o['mask512']>=0).float()[:,None],4,4)[:,0]>=.5).cpu().numpy()
        row={'main':metrics(o['logits'].sigmoid().cpu().numpy(),y)[0],'aux512':metrics(aux,y)[0],
             'local':metrics(o['local'].sigmoid().cpu().numpy(),y)[0],
             'regime_correct':bool((o['meta'][0,0]>=0)==b['full_watermark'][0].bool()),
             'coverage_abs_error':float(abs(o['meta'][0,1].sigmoid()-b['mask128'].float().mean()))}
        if synthetic:
            target=torch.log1p(b['energy'].float());pred=F.softplus(o['energy']);valid=(b['mask128']>0)&(target>0)
            row.update(energy_mae=float((pred-target).abs().mean()),energy_zero_mae=float(target.mean()))
            if valid.any():row.update(energy_positive_mae=float((pred-target)[valid].abs().mean()),energy_positive_zero_mae=float(target[valid].mean()))
        rows.append(row)
    return rows

def summary(rows):
    out={key:summarize([r[key] for r in rows]) for key in ('main','aux512','local')}
    for key in ('regime_correct','coverage_abs_error','energy_mae','energy_zero_mae','energy_positive_mae','energy_positive_zero_mae'):
        valid=[r[key] for r in rows if key in r]
        if valid:out[key]=float(np.mean(valid))
    return out

def run(families=FAMILIES):
    os.chdir(ROOT);torch.set_num_threads(4);assert '3070' in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    path=OUT/'auxiliary_diagnostics.json'
    existing=json.loads(path.read_text()) if path.exists() else {}
    results=existing['families'] if existing.get('fp32_tf32_disabled') else {}
    for family in families:
        if family in results:continue
        folder=OUT/'runs'/family;assert (folder/'completed.json').exists();data=Data(family);model=data.model(True)
        model.load_state_dict(torch.load(folder/'all_information/synthetic.pt',map_location='cpu',weights_only=True)['model'])
        syn=collect(model,data,data.dev,True);real=[];lookup={r['name']:i for i,r in enumerate(data.rows)}
        for fold in (1,2,3):
            f=folder/'all_information'/f'fold{fold}'
            # These are our own locally created checkpoints, including training config.
            model.load_state_dict(torch.load(f/'model.pt',map_location='cpu',weights_only=False)['model'])
            names=json.loads((f/'split.json').read_text())['validation']
            real+=collect(model,data,[lookup[n] for n in names])
        results[family]={'synthetic_dev':summary(syn),'real_oof':summary(real)}
        saved=json.loads((folder/'all_information/completed.json').read_text())['summary']['all']['miou']
        assert abs(results[family]['real_oof']['main']['all']['miou']-saved)<1e-12,'Primary prediction drift during auxiliary audit'
        if family=='pixelseal':
            original=np.load(data.path/'bits.npy',mmap_mode='r');native=np.load(data.path/'bits_native.npy',mmap_mode='r')
            same=original.shape==native.shape and all(np.array_equal(original[j:j+100],native[j:j+100]) for j in range(0,len(original),100))
            results[family]['original_and_native_bit_cache_identical']=bool(same)
        del model,data;torch.cuda.empty_cache()
        print('AUX_AUDIT',family,json.dumps(results[family]),flush=True)
        save(path,{'selection_used':False,'fp32_tf32_disabled':True,'families':results})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--families',nargs='+',choices=FAMILIES,default=FAMILIES);a=p.parse_args();run(a.families)
