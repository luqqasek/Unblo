"""Instrument actual extractors; STN graph, geometry, cache identity, equivariance."""
from common import *
import pandas as pd
import onnx
import onnxruntime as ort
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr

STN_NAMES=['gen_stn/Reshape:0','gen_stn/AddN:0','gen_stn/MatMul_1:0']
FEATURE_NAMES=[f'gen_decoder/dec_conv_{i}/Relu:0' for i in (2,4,6,7)]

def stn_session():
    graph=onnx.load(ROOT/'checkpoints/stegastamp/stega_stamp_logits.onnx')
    names=FEATURE_NAMES+STN_NAMES
    existing={o.name for o in graph.graph.output}
    for name in names:
        if name not in existing:graph.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None))
    path=OUT/'cache/stegastamp_instrumented.onnx';onnx.save(graph,path)
    options=ort.SessionOptions();options.intra_op_num_threads=4
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider']),graph

def inverse_grid(theta,h=128,w=128):
    # Actual ONNX: destination linspace[-1,1], source pixel=(A*u+1)*398/2.
    # Original probability/image grid has align_corners=False center convention.
    y=(torch.arange(h,dtype=torch.float64)+.5)*400/h-.5
    x=(torch.arange(w,dtype=torch.float64)+.5)*400/w-.5
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    src=torch.stack([xx/199-1,yy/199-1],-1)
    a=torch.as_tensor(theta,dtype=torch.float64)
    dst=(src-a[:,2])@torch.linalg.inv(a[:,:2]).T
    # u -> STN output pixel -> align_corners=False normalized sampling coordinates.
    grid=((dst+1)*399/2+.5)*2/400-1
    return grid.float()[None]

def unwarp(p,theta):
    t=torch.from_numpy(p)[None,None]
    return F.grid_sample(t,inverse_grid(theta,*p.shape),mode='bilinear',padding_mode='border',align_corners=False)[0,0].numpy()

def inverse_feature_grid(theta,level,size):
    """Actual conv centers plus adaptive-pooling bin centers; no H/W=stride assumption."""
    from scipy.interpolate import interp1d
    native=[200,100,25,13][level]; stride=[2,4,16,32][level];start=[1,3,15,15][level]
    centers=start+stride*np.arange(native,dtype=float)
    if size<native:
        centers=np.array([centers[int(np.floor(i*native/size)):int(np.ceil((i+1)*native/size))].mean() for i in range(size)])
    yy,xx=np.meshgrid(centers,centers,indexing='ij')
    original=np.stack([xx/199-1,yy/199-1],-1)
    u=(original-theta[:,2])@np.linalg.inv(theta[:,:2]).T
    stn_pixels=(u+1)*399/2
    to_index=interp1d(centers,np.arange(size,dtype=float),fill_value='extrapolate',bounds_error=False)
    grid=(to_index(stn_pixels)+.5)*2/size-1
    return torch.from_numpy(grid.astype(np.float32))[None]

def stn_numpy(image,theta):
    # Match graph's clipped-neighbor bilinear weights, not a guessed grid_sample mode.
    y,x=np.meshgrid(np.linspace(-1,1,400),np.linspace(-1,1,400),indexing='ij')
    pos=np.stack([x,y,np.ones_like(x)],-1)@theta.T
    px=(pos[:,:,0]+1)*199;py=(pos[:,:,1]+1)*199
    x0=np.floor(px).astype(int);y0=np.floor(py).astype(int)
    x1=np.clip(x0+1,0,399);y1=np.clip(y0+1,0,399);x0=np.clip(x0,0,399);y0=np.clip(y0,0,399)
    return ((x1-px)*(y1-py))[:,:,None]*image[y0,x0]+((x1-px)*(py-y0))[:,:,None]*image[y1,x0]+((px-x0)*(y1-py))[:,:,None]*image[y0,x1]+((px-x0)*(py-y0))[:,:,None]*image[y1,x1]

def main():
    ds=cv.model.LocalisationDataset(ROOT/'data','train');checks={};stats=[];shapes=[];equiv=[];execrows=[];decoder=[];stnrows=[];corrections=[]
    mapping={m['image_name']:i for i,m in enumerate(ds.meta)}
    imagehash=[]
    for item in ds:
        assert item['image'].shape==(512,512,3) and item['mask'].shape==(512,512)
        np.testing.assert_array_equal(official.downsample_mask(item['mask']),item['mask128'])
        assert set(np.unique(item['mask128']))<={0,1}
        imagehash.append(hashlib.sha256(item['image'].tobytes()).hexdigest())
    assert len(set(imagehash))==510
    checks.update(images=510,mask_downsample_exact=510,unique_rgb_hashes=510,positive='mask PNG >127; 128-grid 1=watermarked',empty_masks=sum(not v.any() for v in ds.m128.values()))
    # Geometric and channel-order tests, no generated watermark data.
    geom=np.zeros((512,512),np.uint8);geom[40:80,120:200]=1
    expected=np.zeros((128,128),np.uint8);expected[10:20,30:50]=1
    np.testing.assert_array_equal(official.downsample_mask(geom),expected)
    geom[:4,:4]=0;geom[:2,:4]=1;assert official.downsample_mask(geom)[0,0]==1
    rgb=np.zeros((512,512,3),np.uint8);rgb[:,:,0]=255
    t=cv.model.to_tensor(rgb);assert t.shape==(1,3,512,512) and t[:,0].min()==1 and t[:,1:].max()==0
    # Forward/inverse affine landmarks are algebraic and independent of masks.
    a=np.array([[1.02,.03,.04],[-.02,.97,-.05]])
    landmarks=np.array([[-.5,-.4],[.1,.7],[.5,-.1]])
    mapped=landmarks@a[:,:2].T+a[:,2];back=(mapped-a[:,2])@np.linalg.inv(a[:,:2]).T
    assert np.max(abs(back-landmarks))<1e-12
    checks.update(geometry_rectangle=True,half_cell_tie=True,rgb_channel_order=True,affine_landmark_roundtrip_max=float(np.max(abs(back-landmarks))))
    for method in METHODS:
        print('INSTRUMENT',method,flush=True)
        cache=torch.load(cv.model.OUT/f'{method}_features.pt',weights_only=True)
        names=cache['names']; ids=np.linspace(0,169,6).astype(int).tolist()
        wm=cv.model.load_watermark(method,models_dir=ROOT/'checkpoints',device='cpu')
        collected={};counts={};handles=[]
        if method=='stegastamp':
            session,graph=stn_session();layers={};net=None
            producers={o:n for n in graph.graph.node for o in n.output}
            def ancestors(value,seen=None):
                seen=set() if seen is None else seen
                if value in seen:return seen
                seen.add(value)
                if value in producers:
                    for v in producers[value].input:ancestors(v,seen)
                return seen
            assert all('gen_stn/AddN:0' in ancestors(n) for n in FEATURE_NAMES)
            (OUT/'onnx_execution_graph.txt').write_text('\n'.join(f'{n.name}: {n.op_type} {list(n.input)} -> {list(n.output)}' for n in graph.graph.node),encoding='utf-8')
            checks['stn_active_ancestor_all_feature_levels']=True
        else:
            net=wm.model if method=='pixelseal' else wm.net
            # Observe wrapper defaults BEFORE applying the original extractor freeze.
            initial_train=[n for n,m in net.named_modules() if m.training]
            net.requires_grad_(False)
            layers={f'stage{i}':v for i,v in enumerate(wm.det.convnext.stages)} if method=='pixelseal' else {f'layer{i}':getattr(net,f'layer{i}') for i in range(1,5)}
            def hook(name):
                def capture(m,a,o):collected[name]=o.detach().clone();counts[name]=counts.get(name,0)+1
                return capture
            for name,layer in layers.items():handles.append(layer.register_forward_hook(hook(name)))
            buffers={k:v.clone() for k,v in net.named_buffers()}
            (OUT/f'{method}_module_tree.txt').write_text(str(net),encoding='utf-8')
            execrows.append(dict(method=method,class_name=type(net).__module__+'.'+type(net).__name__,training_modules=initial_train,
                frozen=all(not p.requires_grad for p in net.parameters()),layers={k:type(v).__module__+'.'+type(v).__name__ for k,v in layers.items()}))
        def forward(x):
            if method=='stegastamp':
                values=session.run(FEATURE_NAMES+[wm.LOGITS]+STN_NAMES,{'image':wm._prep(x),'secret':np.zeros((len(x),100),np.float32)})
                return [torch.from_numpy(v) for v in values[:4]],values[4],values[5:]
            collected.clear();logits=wm.logits(x)
            return [collected[k] for k in layers],logits.numpy(),[]
        # StegaStamp all-image transformation distribution and message logits; others all-image decoder run too.
        for i,name in enumerate(names):
            item=ds[mapping[name]];x=cv.model.to_tensor(item['image'])
            with torch.inference_mode():features,logits,extra=forward(x)
            bits=item['message'];acc=float(((logits[0]>0)==bits).mean());assert abs(acc-cache['bit_accuracy'][i])<1e-7
            signed=logits[0]*(2*bits.astype(float)-1)
            confidence=float((1/(1+np.exp(-np.clip(signed,-50,50)))).mean())
            decoder.append(dict(method=method,image_id=name,bit_accuracy=acc,message_log_likelihood=float(-np.logaddexp(0,-signed).mean()),expected_bit_probability=confidence,
                raw_logit_abs=float(abs(logits).mean()),ecc=False))
            if method=='stegastamp':
                theta,warped,grid=extra;theta=theta[0];warped=warped[0].reshape(400,400,3)
                scales=np.linalg.svd(theta[:,:2],compute_uv=False);corners=np.array([[-1,-1],[1,-1],[-1,1],[1,1]])
                disp=(corners@theta[:,:2].T+theta[:,2]-corners)*256
                stnrows.append(dict(image_id=name,a00=theta[0,0],a01=theta[0,1],tx=theta[0,2],a10=theta[1,0],a11=theta[1,1],ty=theta[1,2],
                    rotation_deg=float(np.degrees(np.arctan2(theta[1,0],theta[0,0]))),scale_max=scales[0],scale_min=scales[1],
                    shear_column_cos=float(np.dot(theta[:,0],theta[:,1])/(np.linalg.norm(theta[:,0])*np.linalg.norm(theta[:,1]))),
                    determinant=float(np.linalg.det(theta[:,:2])),max_corner_displacement_512=float(np.linalg.norm(disp,axis=1).max()),translation_x_512=theta[0,2]*256,translation_y_512=theta[1,2]*256))
                if i in ids:
                    prep=wm._prep(x)[0];reconstructed=stn_numpy(prep,theta)
                    checks[f'stn_sampler_max_error_{name}']=float(np.max(abs(reconstructed-warped)))
                    assert checks[f'stn_sampler_max_error_{name}']<.003
                    fig,ax=plt.subplots(1,3,figsize=(11,4));ax[0].imshow(prep);ax[1].imshow(np.clip(warped,0,1))
                    gridv=grid[0].reshape(2,400,400)
                    for k in range(0,400,25):ax[2].plot(gridv[0,k,:],gridv[1,k,:],color='navy',lw=.6);ax[2].plot(gridv[0,:,k],gridv[1,:,k],color='navy',lw=.6)
                    for aa,title in zip(ax,['Original after 400 resize','Actual STN output','Output-to-input grid']):aa.set_title(title)
                    ax[2].invert_yaxis();fig.tight_layout();fig.savefig(OUT/'figures'/f'stn_{i}.png',dpi=120);plt.close(fig)
            if i not in ids:continue
            for k,f in enumerate(features):
                capped=F.adaptive_avg_pool2d(f,(min(32,f.shape[-2]),min(32,f.shape[-1])))[0]
                err=float((capped-cache['features'][k][i]).abs().max());assert err<1e-5
                v=f.numpy();stats.append(dict(method=method,image_id=name,layer=cache['layers'][k],minimum=float(v.min()),maximum=float(v.max()),mean=float(v.mean()),std=float(v.std()),spatial_variance=float(v.var(axis=(-2,-1)).mean()),zero_fraction=float((v==0).mean()),finite=bool(np.isfinite(v).all()),cache_max_error=err))
                if i==ids[0]:shapes.append(dict(method=method,feature_layer=cache['layers'][k],shape=str(list(f.shape)),channels=f.shape[1],native_height=f.shape[2],native_width=f.shape[3],grid_spacing_512=512/f.shape[2],cache_height=cache['features'][k].shape[2],coordinate_system='after active STN' if method=='stegastamp' else 'resized full image',rf_note='see architecture audit; grid spacing is not receptive field'))
            # Visualization: mean and most spatially variable channel; descriptive, no mask-based choice.
            fig,ax=plt.subplots(2,4,figsize=(12,6))
            for k,f in enumerate(features):
                v=f[0].numpy();channel=v.var(axis=(1,2)).argmax();ax[0,k].imshow(v.mean(0));ax[1,k].imshow(v[channel]);ax[0,k].set_title(cache['layers'][k]);ax[1,k].set_title(f'channel {channel}')
            for aa in ax.ravel():aa.axis('off')
            fig.tight_layout();fig.savefig(OUT/'figures'/f'features_{method}_{i}.png',dpi=110);plt.close(fig)
            # +16 original pixels = 4 target cells. Bilinear reference handles fractional native stride.
            shifted=F.pad(x[:,:,:,:-16],(16,0,0,0),mode='replicate')
            with torch.inference_mode():moved,_,_=forward(shifted)
            for k,(orig,new) in enumerate(zip(features,moved)):
                h,w=orig.shape[-2:]; yy,xx=torch.meshgrid((torch.arange(h)+.5)*2/h-1,(torch.arange(w)+.5)*2/w-1,indexing='ij')
                grid=torch.stack([xx-2*16/512,yy],-1)[None]
                ref=F.grid_sample(orig,grid,align_corners=False,padding_mode='border')
                # Remove 20% border; global receptive fields may still see it.
                b=max(1,int(w*.2));a=ref[:,:,b:-b,b:-b].flatten().numpy();v=new[:,:,b:-b,b:-b].flatten().numpy()
                equiv.append(dict(method=method,image_id=name,layer=cache['layers'][k],shift_original_px=16,correlation=float(np.corrcoef(a,v)[0,1]),relative_rmse=float(np.sqrt(np.mean((a-v)**2))/(np.std(a)+1e-12)),margin_fraction=.2))
            cohort=load_cohort(method,'stable' if method=='stegastamp' else 'original')
            ck=torch.load(cohort['checkpoints'][cohort['fold'][i]-1],weights_only=True);head=cv.model.Head(ck['channels']);head.load_state_dict(ck['state_dict']);head.eval()
            capped=[F.adaptive_avg_pool2d(f,(min(32,f.shape[-2]),min(32,f.shape[-1]))) for f in moved]
            with torch.inference_mode():moved_prob=head(capped).sigmoid()[0].numpy()
            a=cohort['prob'][i,24:-24,20:-28].ravel();v=moved_prob[24:-24,24:-24].ravel()
            equiv.append(dict(method=method,image_id=name,layer='OOF_prediction',shift_original_px=16,correlation=float(np.corrcoef(a,v)[0,1]),relative_rmse=float(np.sqrt(np.mean((a-v)**2))/(np.std(a)+1e-12)),margin_fraction=.1875))
        if net is not None:
            assert all(torch.equal(v,buffers[k]) for k,v in net.named_buffers());checks[method+'_buffers_unchanged']=True
            checks[method+'_hook_calls']=counts
            for h in handles:h.remove()
        print('DONE',method,flush=True)
    theta_by_name={r['image_id']:np.array([[r['a00'],r['a01'],r['tx']],[r['a10'],r['a11'],r['ty']]]) for r in stnrows}
    for variant in ('original','stable'):
        c=load_cohort('stegastamp',variant)
        for i,name in enumerate(c['names']):
            p=c['prob'][i];g=c['gt'][i];t=c['threshold'][i];corrected=unwarp(p,theta_by_name[name])
            before=official.miou_image(p>=t,g);after=official.miou_image(corrected>=t,g)
            corrections.append(dict(image_id=name,variant=variant,before=before,after=after,delta=after-before,kind='inverse actual STN on same frozen-head probability, no retraining'))
    for name,rows in [('feature_shapes',shapes),('feature_statistics',stats),('equivariance',equiv),('decoder_metrics',decoder),('stn_transforms',stnrows),('stn_corrections',corrections)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    save_json(OUT/'execution_checks.json',dict(checks=checks,execution=execrows,stn_checkpoint_sha256=sha(ROOT/'checkpoints/stegastamp/stega_stamp_logits.onnx')))
    print('STN correction',pd.DataFrame(corrections).groupby('variant')[['before','after','delta']].mean().to_string(),flush=True)

if __name__=='__main__':main()
