"""Gradient receptive-field estimate on actual PixelSeal/TrustMark graphs."""
from common import *
import pandas as pd
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ds=cv.model.LocalisationDataset(ROOT/'data','train');lookup={m['image_name']:i for i,m in enumerate(ds.meta)};rows=[]
    for method in ('pixelseal','trustmark'):
        wm=cv.model.load_watermark(method,models_dir=ROOT/'checkpoints',device='cpu');net=wm.model if method=='pixelseal' else wm.net;net.requires_grad_(False)
        layers=list(wm.det.convnext.stages) if method=='pixelseal' else [getattr(net,f'layer{i}') for i in range(1,5)]
        names=load_cohort(method)['names'];captured={}
        handles=[layer.register_forward_hook(lambda m,a,o,k=k:captured.__setitem__(k,o)) for k,layer in enumerate(layers)]
        for i in (0,84,169):
            x=cv.model.to_tensor(ds[lookup[names[i]]]['image']).requires_grad_()
            if method=='pixelseal':wm.det(wm._resize(x))
            else:wm.net(wm._prep(x))
            fig,ax=plt.subplots(1,4,figsize=(12,3))
            for k,f in captured.items():
                # Mean squared activation at center across channels; conditional, not full RF proof.
                objective=f[0,:,f.shape[2]//2,f.shape[3]//2].square().mean()
                grad=torch.autograd.grad(objective,x,retain_graph=True)[0].abs().sum(1)[0].numpy()
                mass=grad/grad.sum();y,xg=np.indices(mass.shape);cy=float((mass*y).sum());cx=float((mass*xg).sum());radius=np.sqrt((xg-cx)**2+(y-cy)**2)
                order=np.argsort(radius.ravel());cum=np.cumsum(mass.ravel()[order]);r90=float(radius.ravel()[order[np.searchsorted(cum,.9)]])
                rows.append(dict(method=method,image_id=names[i],level=k+1,gradient_mass_radius90_px=r90,centroid_x=cx,centroid_y=cy,nonzero_fraction=float((grad>grad.max()*1e-6).mean()),definition='center activation squared norm, abs input gradient mass; 3 images only'))
                ax[k].imshow(np.log10(grad+1e-12),cmap='magma');ax[k].set_title(f'Level {k+1}; r90={r90:.1f}');ax[k].axis('off')
            fig.tight_layout();fig.savefig(OUT/'figures'/f'rf_{method}_{i}.png',dpi=110);plt.close(fig)
        for h in handles:h.remove()
    pd.DataFrame(rows).to_csv(OUT/'effective_receptive_fields.csv',index=False)

if __name__=='__main__':main()
