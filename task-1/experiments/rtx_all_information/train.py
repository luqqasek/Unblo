"""Matched all-data comparison on RTX3070; complete coverage and outer split assertions."""
import argparse, copy, json, os, time
from pathlib import Path
import numpy as np
import torch
from .prepare import ROOT,HERE,OUT,FAMILIES,save,sha
from .model import AllInformation,objective

class Data:
    def __init__(self,family):
        self.family=family;self.path=OUT/'data'/family
        assert (self.path/'complete.json').exists()
        self.rows=json.loads((self.path/'records.json').read_text())
        self.arrays={p.stem:np.load(p,mmap_mode='r') for p in self.path.glob('*.npy')}
        if family=='pixelseal':
            assert (self.path/'native_bits_complete.json').exists()
            self.arrays['bits']=self.arrays.pop('bits_native')
        self.train=np.array([i for i,r in enumerate(self.rows) if r['partition']=='train'])
        self.dev=np.array([i for i,r in enumerate(self.rows) if r['partition']=='validation'])
        self.real=np.array([i for i,r in enumerate(self.rows) if r['partition']=='real'])
        assert len(self.train)==4000 and len(self.dev)==200 and len(self.real)==170
    def batch(self,ids):
        ids=list(map(int,ids))
        result={k:torch.from_numpy(np.array(a[ids])).cuda(non_blocking=True) for k,a in self.arrays.items() if not k.startswith('feature')}
        result['features']=[torch.from_numpy(np.array(self.arrays[f'feature{k}'][ids])).cuda(non_blocking=True) for k in range(4)]
        result['full_watermark']=torch.tensor([self.rows[i]['full_watermark'] for i in ids],device='cuda',dtype=torch.float32)
        result['synthetic']=torch.tensor([self.rows[i]['synthetic'] for i in ids],device='cuda')
        return result
    def model(self,full):
        return AllInformation([self.arrays[f'feature{k}'].shape[1] for k in range(4)],self.arrays['message'].shape[1],self.arrays['bits'].shape[1],self.arrays['scalar'].shape[1],full).cuda()

def metrics(p,y):
    t=y.astype(bool);pred=p>=.5
    tp=np.sum(pred&t,axis=(1,2));fp=np.sum(pred&~t,axis=(1,2));fn=np.sum(~pred&t,axis=(1,2));tn=np.sum(~pred&~t,axis=(1,2))
    fg=np.divide(tp,tp+fp+fn,out=np.ones(len(tp),float),where=tp+fp+fn>0)
    clean=np.divide(tn,tn+fp+fn,out=np.ones(len(tp),float),where=tn+fp+fn>0)
    return [dict(miou=float((a+b)/2),fg_iou=float(a),clean_iou=float(b),true_area=float(t[i].mean()),pred_area=float(pred[i].mean()),fg_miss=bool(tp[i]==0 and t[i].any()),clean_miss=bool(tn[i]==0 and (~t[i]).any())) for i,(a,b) in enumerate(zip(fg,clean))]

def summarize(rows):
    groups={'all':rows,'fg_le1':[r for r in rows if 0<r['true_area']<=.01], 'fg_1to5':[r for r in rows if .01<r['true_area']<=.05], 'fg_5to20':[r for r in rows if .05<r['true_area']<=.2], 'clean_le1':[r for r in rows if 0<1-r['true_area']<=.01], 'clean_1to5':[r for r in rows if .01<1-r['true_area']<=.05], 'clean_5to20':[r for r in rows if .05<1-r['true_area']<=.2]}
    groups.update({label:[r for r in rows if r.get('full_watermark')==value] for label,value in [('initial_full',True),('initial_local',False)]})
    return {g:{'n':len(rr),**{k:float(np.mean([r[k] for r in rr])) if rr else None for k in ['miou','fg_iou','clean_iou']},'fg_miss':sum(r['fg_miss'] for r in rr),'clean_miss':sum(r['clean_miss'] for r in rr)} for g,rr in groups.items()}

@torch.inference_mode()
def evaluate(model,data,ids,dest,wrong=False):
    model.eval();probs=[];rows=[]
    for i in ids:
        b=data.batch([i]);p=model(b,wrong_message=wrong)['logits'].sigmoid()[0].cpu().numpy()
        r=metrics(p[None],b['mask128'].cpu().numpy())[0];r['image_name']=data.rows[int(i)]['name'];r['full_watermark']=bool(data.rows[int(i)]['full_watermark']);rows.append(r);probs.append(p)
    save(dest.with_suffix('.json'),{'summary':summarize(rows),'per_image':rows})
    np.savez_compressed(dest.with_suffix('.npz'),names=[r['image_name'] for r in rows],probabilities=probs,targets=np.array(data.arrays['mask128'][list(ids)]))
    model.train();return rows

def step(model,opt,data,ids,micro):
    opt.zero_grad(set_to_none=True);logs={}
    for j in range(0,len(ids),micro):
        b=data.batch(ids[j:j+micro]);out=model(b);loss,terms=objective(model,b,out);weight=len(ids[j:j+micro])/len(ids)
        (loss*weight).backward()
        for k,v in terms.items():logs[k]=logs.get(k,0)+float(v.detach())*weight
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(norm),logs
    opt.step();return logs

def run(family,micro=2):
    os.chdir(ROOT);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert '3070' in torch.cuda.get_device_name(0)
    if family=='pixelseal':
        from .spatial_bits import prepare_native_bits
        prepare_native_bits()
    torch.cuda.reset_peak_memory_stats()
    data=Data(family);start=time.monotonic();folder=OUT/'runs'/family;folder.mkdir(parents=True,exist_ok=True)
    config={'family':family,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'seed':42,'synthetic_epochs':2,'real_epochs':20,'effective_batch':8,'microbatch':micro,'threshold':.5,'synthetic_train':4000,'synthetic_dev':200,'real':170,'code_sha256':sha(__file__),'model_code_sha256':sha(HERE/'model.py')}
    config.update(bit_map_shape=list(data.arrays['bits'].shape[1:]),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,split_sha256=sha(ROOT/'experiments/native_localizer/reconstructed_splits.json'))
    save(folder/'config.json',config)
    splits=json.loads((ROOT/'experiments/native_localizer/reconstructed_splits.json').read_text())['families'][family]
    lookup={data.rows[i]['name']:int(i) for i in data.real}
    for full in [False,True]:
        arm='all_information' if full else 'control';dest=folder/arm;dest.mkdir(exist_ok=True)
        if (dest/'completed.json').exists():continue
        torch.manual_seed(42);np.random.seed(42);model=data.model(full);model.train();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        pretrain=dest/'synthetic.pt'
        if pretrain.exists():model.load_state_dict(torch.load(pretrain,map_location='cpu',weights_only=True)['model'])
        else:
            seen=set();updates=0;began=time.monotonic()
            for epoch in range(2):
                order=np.random.default_rng(4242+epoch).permutation(data.train)
                for j in range(0,len(order),8):
                    ids=order[j:j+8];logs=step(model,opt,data,ids,micro);seen.update(map(int,ids));updates+=1
                    if updates%25==0:
                        elapsed=time.monotonic()-began
                        print('TRAIN',family,arm,'synthetic',epoch+1,updates,'seconds',round(elapsed,1),'eta',round(elapsed/updates*(1000-updates),1),'loss',logs,'peakGB',round(torch.cuda.max_memory_allocated()/1e9,2),flush=True)
            assert seen==set(map(int,data.train))
            torch.save({'model':model.state_dict()},pretrain)
            save(dest/'synthetic_coverage.json',{'unique':len(seen),'ids':sorted(seen),'epochs':2,'views':8000,'updates':updates})
        evaluate(model,data,data.dev,dest/'synthetic_dev')
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};del model,opt;torch.cuda.empty_cache()
        allrows=[];seen_valid=set()
        for fold,split in enumerate(splits,1):
            fdir=dest/f'fold{fold}';fdir.mkdir(exist_ok=True)
            valid=[lookup[n] for n in split['validation']];training=[int(i) for i in data.real if i not in set(valid)]
            assert not set(training)&set(valid) and len(training)+len(valid)==170
            assert {data.rows[i]['name'] for i in training}==set(split['train'])
            assert not seen_valid&set(valid);seen_valid.update(valid)
            if (fdir/'completed.json').exists():allrows.extend(json.loads((fdir/'validation.json').read_text())['per_image']);continue
            torch.manual_seed(42);model=data.model(full);model.load_state_dict(initial);model.train();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
            save(fdir/'split.json',{'train':[data.rows[i]['name'] for i in training],'validation':[data.rows[i]['name'] for i in valid]})
            counts={i:0 for i in training};updates=0;began=time.monotonic()
            for epoch in range(20):
                rng=np.random.default_rng(7000+epoch);order=rng.permutation(training);replay=rng.permutation(data.train)
                for j in range(0,len(order),4):
                    real=list(order[j:j+4]);syn=list(replay[j:j+4]);ids=real+syn
                    logs=step(model,opt,data,ids,micro);updates+=1
                    for i in real:counts[int(i)]+=1
                if epoch%5==0 or epoch==19:print('TRAIN',family,arm,'fold',fold,'epoch',epoch+1,'seconds',round(time.monotonic()-began,1),'loss',logs,flush=True)
            assert set(counts)==set(training) and min(counts.values())==max(counts.values())==20
            rows=evaluate(model,data,valid,fdir/'validation');allrows.extend(rows)
            if full:evaluate(model,data,valid,fdir/'wrong_message',wrong=True)
            torch.save({'model':model.state_dict(),'config':config},fdir/'model.pt')
            save(fdir/'completed.json',{'summary':summarize(rows),'train_unique':len(counts),'train_visits_per_image':20,'validation_unique':len(valid),'updates':updates,'seconds':time.monotonic()-began,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
            print('FOLD_RESULT',family,arm,fold,json.dumps(summarize(rows)),flush=True)
            del model,opt;torch.cuda.empty_cache()
        assert seen_valid==set(map(int,data.real)) and len(allrows)==170 and len({r['image_name'] for r in allrows})==170
        save(dest/'completed.json',{'summary':summarize(allrows),'per_image':allrows,'seconds_total_so_far':time.monotonic()-start})
    control=json.loads((folder/'control/completed.json').read_text());full=json.loads((folder/'all_information/completed.json').read_text())
    c={r['image_name']:r for r in control['per_image']};f={r['image_name']:r for r in full['per_image']};names=sorted(c);assert set(c)==set(f)
    delta=np.array([f[n]['miou']-c[n]['miou'] for n in names]);rng=np.random.default_rng(42)
    result={'family':family,'control':control['summary'],'all_information':full['summary'],'delta':float(delta.mean()),'conditional_bootstrap95':np.quantile(delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1),[.025,.975]).tolist(),'improved':int((delta>1e-8).sum()),'worsened':int((delta<-1e-8).sum()),'seconds':time.monotonic()-start,'gpu':torch.cuda.get_device_name(0),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    save(folder/'completed.json',result);print('FAMILY_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=FAMILIES,required=True);p.add_argument('--microbatch',type=int,default=2);a=p.parse_args();run(a.family,a.microbatch)
