"""Frozen pretrained decoder evidence; never use masks to compute features."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'frozen_extractor'))
import cross_validate as cv
OUT=HERE/'results'


def spatial_evidence(logits,messages):
    signed=logits*(2*messages[:,:,None,None]-1)
    probability=signed.sigmoid().mean(1)
    hard=(signed>0).float().mean(1)
    relative=signed.mean(1)/(logits.abs().mean(1)+1e-6)
    global_match=(logits.mean((2,3))*(2*messages-1)).sigmoid().mean(1)
    return torch.stack([probability,hard,relative,global_match[:,None,None].expand_as(probability)],1)


def occlusion_logits(wm,image):
    """Original image followed by 16 mean-color occlusions in row-major order."""
    assert image.shape==(3,512,512)
    batch=image[None].repeat(17,1,1,1)
    fill=image.mean((1,2),keepdim=True)
    for a in range(4):
        for b in range(4):
            batch[1+4*a+b,:,a*128:(a+1)*128,b*128:(b+1)*128]=fill
    return wm.logits(batch,bs=4)


def occlusion_bits(logits,message):
    assert logits.shape==(17,100) and message.shape==(100,)
    score=F.logsigmoid(logits*(2*message-1))
    return ((score[0]-score[1:])/5).tanh().T.reshape(100,4,4)


def prepare(method):
    path=OUT/f'{method}_evidence.pt'
    if path.exists(): return torch.load(path,weights_only=True)
    ds=cv.model.LocalisationDataset(cv.model.ROOT/'data','train')
    cache=cv.model.extract(method,ds); names=cache['names']
    meta={m['image_name']:m for m in ds.meta}
    messages=torch.tensor([[int(c) for c in meta[n]['message']] for n in names],dtype=torch.float32)
    wrong=messages.roll(1,0)
    wm=cv.model.load_watermark(method,cv.model.ROOT/'checkpoints','cpu')
    real=[]; shuffled=[]; global_acc=[]; verification_error=[]
    if method in ('pixelseal','trustmark'):
        with torch.inference_mode():
            for j in range(0,len(names),8):
                latent=cache['features'][-1][j:j+8]
                if method=='pixelseal':
                    latent=wm.det.pixel_decoder.output_upscaling(latent)
                    layer=wm.det.pixel_decoder.linear
                    logits=torch.einsum('bchw,kc->bkhw',latent,layer.weight)+layer.bias[None,:,None,None]
                    logits=logits[:,1:]
                else:
                    layer=wm.net.fc
                    logits=torch.einsum('bchw,kc->bkhw',latent,layer.weight)+layer.bias[None,:,None,None]
                real.append(spatial_evidence(logits,messages[j:j+8]))
                shuffled.append(spatial_evidence(logits,wrong[j:j+8]))
                global_acc.extend(((logits.mean((2,3))>0)==messages[j:j+8]).float().mean(1).tolist())
                # Verify pretrained local projection against real image-level inference.
                for k in range(len(logits)):
                    rgb=np.asarray(Image.open(cv.model.ROOT/'data/train/images'/f'{names[j+k]}.png').convert('RGB'))
                    native=wm.logits(cv.model.to_tensor(rgb))[0]
                    error=float((native-logits[k].mean((1,2))).abs().max())
                    verification_error.append(error)
                    torch.testing.assert_close(native,logits[k].mean((1,2)),rtol=1e-4,atol=1e-4)
                print(method,'projection',j+len(logits),flush=True)
        real=torch.cat(real); shuffled=torch.cat(shuffled)
    else:
        # Occlusion is in original image coordinates, avoiding false spatial claims
        # about StegaStamp's spatial transformer and flattened dense layers.
        with torch.inference_mode():
            for i,name in enumerate(names):
                checkpoint=OUT/'occlusion'/f'{name}.pt'
                if checkpoint.exists():
                    logits=torch.load(checkpoint,weights_only=True)
                else:
                    rgb=np.asarray(Image.open(cv.model.ROOT/'data/train/images'/f'{name}.png').convert('RGB'))
                    image=cv.model.to_tensor(rgb)[0]
                    logits=occlusion_logits(wm,image)
                    checkpoint.parent.mkdir(parents=True,exist_ok=True); torch.save(logits,checkpoint)
                global_acc.append(float(((logits[0]>0)==messages[i]).float().mean()))
                for msg,dest in [(messages[i],real),(wrong[i],shuffled)]:
                    signed=logits*(2*msg-1)
                    prob=signed.sigmoid().mean(1); score=F.logsigmoid(signed).mean(1)
                    hard=(signed>0).float().mean(1)
                    dest.append(torch.stack([(prob[0]-prob[1:]).reshape(4,4),
                        ((score[0]-score[1:])/5).clamp(-1,1).reshape(4,4),
                        prob[0].expand(4,4),hard[1:].reshape(4,4)]))
                if (i+1)%10==0: print('stegastamp occlusion',i+1,'/170',flush=True)
        real=torch.stack(real); shuffled=torch.stack(shuffled)
    assert torch.isfinite(real).all() and torch.isfinite(shuffled).all()
    result=dict(names=names,real=real,wrong=shuffled,global_bit_accuracy=global_acc,
                projection_max_error=max(verification_error) if verification_error else None,
                kind='pretrained spatial projection' if method!='stegastamp' else '4x4 occlusion sensitivity')
    torch.save(result,path)
    print(method,'global bit accuracy',np.mean(global_acc),'shape',list(real.shape),flush=True)
    return result


def prepare_bits(method,evidence):
    path=OUT/f'{method}_perbit.pt'
    if path.exists(): return {**evidence,**torch.load(path,weights_only=True)}
    ds=cv.model.LocalisationDataset(cv.model.ROOT/'data','train')
    cache=cv.model.extract(method,ds); names=cache['names']
    meta={m['image_name']:m for m in ds.meta}
    messages=torch.tensor([[int(c) for c in meta[n]['message']] for n in names],dtype=torch.float32)
    real=[]; wrong=[]
    with torch.inference_mode():
        if method!='stegastamp':
            wm=cv.model.load_watermark(method,cv.model.ROOT/'checkpoints','cpu')
            for j in range(0,len(names),8):
                latent=cache['features'][-1][j:j+8]
                if method=='pixelseal':
                    latent=wm.det.pixel_decoder.output_upscaling(latent); layer=wm.det.pixel_decoder.linear
                else: layer=wm.net.fc
                logits=torch.einsum('bchw,kc->bkhw',latent,layer.weight)+layer.bias[None,:,None,None]
                if method=='pixelseal': logits=logits[:,1:]
                for msg,dest in [(messages,real),(messages.roll(1,0),wrong)]:
                    dest.append((logits*(2*msg[j:j+8,:,None,None]-1)).sigmoid())
            real=torch.cat(real); wrong=torch.cat(wrong)
        else:
            for i,name in enumerate(names):
                logits=torch.load(OUT/'occlusion'/f'{name}.pt',weights_only=True)
                for msg,dest in [(messages,real),(messages.roll(1,0),wrong)]:
                    dest.append(occlusion_bits(logits,msg[i]))
            real=torch.stack(real); wrong=torch.stack(wrong)
    result=dict(perbit=real,wrong_perbit=wrong)
    torch.save(result,path)
    return {**evidence,**result}


if __name__=='__main__':
    torch.set_num_threads(4); OUT.mkdir(parents=True,exist_ok=True)
    for method in cv.METHODS: prepare(method)
