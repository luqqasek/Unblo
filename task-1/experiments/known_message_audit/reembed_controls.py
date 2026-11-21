"""Eight matched clean/marked COCO covers per family: positive-control diagnosis."""
import io
import json
import zipfile
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import gaussian_blur
from reembed_probe import ncc
from probe import ROOT,OUT,save,metrics,load_watermark


def tensor(a):return torch.tensor(np.array(a),device='cuda:0').permute(2,0,1)[None].float()/255


def main():
    torch.set_num_threads(2)
    archive=ROOT/'experiments/synthetic_pretrain_2k/results/sources/val2017.zip'
    with zipfile.ZipFile(archive) as z:
        files=sorted(n for n in z.namelist() if n.endswith('.jpg'))[:8]
        covers=[]
        for filename in files:
            im=Image.open(io.BytesIO(z.read(filename))).convert('RGB');w,h=im.size;s=min(w,h)
            covers.append(np.array(im.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2)).resize((512,512),Image.Resampling.BICUBIC)))
    mask=np.ones((512,512),bool);mask[128:384,128:384]=False
    target=mask.reshape(128,4,128,4).mean((1,3))>.5
    for method in ('pixelseal','trustmark','stegastamp'):
        path=OUT/f'reembed_controls_{method}.json'
        if path.exists():continue
        wm=load_watermark(method,ROOT/'checkpoints','cpu' if method=='stegastamp' else 'cuda:0')
        rows=[]
        for i,cover in enumerate(covers):
            rng=np.random.default_rng(42100+i)
            bits=rng.integers(0,2,wm.nbits);wrong=rng.integers(0,2,wm.nbits)
            marked=wm.embed(cover,bits)
            partial=np.where(mask[:,:,None],marked,cover).astype(np.uint8)
            maps={};stats={}
            for label,image in [('clean',cover),('partial',partial)]:
                x=tensor(image);hp=x-gaussian_blur(x,[19,19],[3.,3.])
                residual=[]
                for b in (bits,wrong):
                    residual.append(wm.embed(image,b).astype(np.float32)-wm.embed(image,1-b).astype(np.float32))
                r=torch.tensor(np.stack(residual),device='cuda:0').permute(0,3,1,2)/255
                if method=='pixelseal':
                    weights=torch.tensor([.299,.587,.114],device='cuda:0')[None,:,None,None]
                    hp=(hp*weights).sum(1,keepdim=True);r=(r*weights).sum(1,keepdim=True)
                scores=ncc(hp,r,32).cpu().numpy()
                stats[label]=dict(true_mean=float(scores[0].mean()),wrong_mean=float(scores[1].mean()))
                if label=='partial':
                    maps={'ncc_true':scores[0],'ncc_wrong':scores[1]}
            decoded=wm.logits(tensor(marked)).cpu().numpy()[0]>0
            rows.append(dict(source=files[i],full_marked_bit_accuracy=float((decoded==bits).mean()),
                             scores=stats,metrics=metrics(maps,target)))
        save(path,dict(method=method,n=8,geometry='Full-frame embedding, central 256x256 clean hole; original cover and message known.',rows=rows))
        print(method,'controls',np.mean([r['metrics']['ncc_true']['auc'] for r in rows]),flush=True)
        del wm;torch.cuda.empty_cache()


if __name__=='__main__':main()
