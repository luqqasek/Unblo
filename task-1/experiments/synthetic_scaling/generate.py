"""Generate real encodings for all families, with a nested 2k/4k independent-cover pool."""
import argparse
import io
import time
import zipfile
from PIL import Image
import torch.nn.functional as F
from scaling_common import *

def bbox(region, max_aspect=None):
    y, x = np.where(region)
    x0, x1, y0, y1 = int(x.min()), int(x.max()) + 1, int(y.min()), int(y.max()) + 1
    if max_aspect:
        w, h = x1 - x0, y1 - y0
        nh = max(h, int(np.ceil(w / max_aspect))); nw = max(w, int(np.ceil(h / max_aspect)))
        x0 = max(0, min(512 - nw, (x0 + x1 - nw) // 2))
        y0 = max(0, min(512 - nh, (y0 + y1 - nh) // 2))
        x1, y1 = x0 + nw, y0 + nh
        assert max(nw / nh, nh / nw) <= max_aspect
    return [x0, y0, x1, y1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--method', choices=METHODS, required=True)
    ap.add_argument('--device', default='cpu'); args = ap.parse_args(); setup(args.device)
    method = args.method; dest = OUT / 'data' / method; dest.mkdir(parents=True, exist_ok=True)
    (dest / 'chunks').mkdir(exist_ok=True); (dest / 'examples').mkdir(exist_ok=True)
    source = module('scaling_source_generator', OLD / 'prepare.py')
    with zipfile.ZipFile(OLD / 'results/sources/annotations_trainval2017.zip') as archive:
        coco = json.loads(archive.read('annotations/instances_val2017.json'))
    annotations = {}
    for a in coco['annotations']: annotations.setdefault(a['image_id'], []).append(a)
    records = sorted(coco['images'], key=lambda r: r['id'])
    order = np.random.default_rng(20261121).permutation(len(records))
    old_manifest = json.loads((OLD / 'results/manifest.json').read_text())
    # The original pool rejected no sources; prove this before extending its deterministic indexing.
    assert not json.loads((OLD / 'results/source_quality.json').read_text())['rejected']
    assert [records[i]['id'] for i in order[:2200]] == [r['coco_id'] for r in old_manifest]
    real_images = [np.asarray(Image.open(p).convert('RGB')) for p in sorted((ROOT / 'data/train/images').glob('*.png'))]
    real_dhash = np.stack([source.dhash(a) for a in real_images])
    real_hashes = {hashlib.sha256(a.tobytes()).hexdigest() for a in real_images}; del real_images
    from wmloc.watermarks import load_watermark
    config = dict(method=method, count=4200, nested_train=[2000,4000], heldout=200,
                  seed=20261121, device=args.device, torch=str(torch.__version__), code_sha256=sha(__file__),
                  old_manifest_sha256=sha(OLD / 'results/manifest.json'), source_generator_sha256=sha(OLD / 'prepare.py'),
                  watermark_wrapper_sha256=sha(ROOT / 'wmloc/watermarks.py'), dtype='float32',
                  trustmark_max_encoder_aspect=2, full_train_preserved=sha(OLD / 'results/full_train/submission.pt'))
    if (dest / 'config.json').exists(): assert json.loads((dest / 'config.json').read_text()) == config
    else: save(dest / 'config.json', config)
    wm = load_watermark(method, models_dir=ROOT / 'checkpoints', device=args.device)
    handles = []; collected = {}
    if method == 'stegastamp':
        import onnxruntime as ort
        options = ort.SessionOptions(); options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(OLD / 'results/cache/features.onnx'), options, providers=['CPUExecutionProvider'])
        wm.sess = session
    else:
        if method == 'pixelseal':
            wm.model.requires_grad_(False).eval()
            layers = list(wm.det.convnext.stages)
        else:
            wm.net.requires_grad_(False).eval()
            wm.tm.encoder.eval().requires_grad_(False)
            layers = [getattr(wm.net, f'layer{i}') for i in range(1,5)]
        for k, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(lambda m,a,o,key=k: collected.__setitem__(key,o.detach())))
    def features(image, message):
        tensor = source.to_tensor(image)
        if method == 'stegastamp':
            values = session.run(source.FEATURE_NAMES + ['gen_stn/Reshape:0', wm.LOGITS],
                                 {'image': wm._prep(tensor), 'secret': np.zeros((1,100),np.float32)})
            fs = [torch.from_numpy(v) for v in values[:4]]; theta = values[-2][0]; logits = values[-1][0]
        else:
            with torch.inference_mode(): logits = wm.logits(tensor)[0].cpu().numpy()
            assert len(collected) == 4
            fs = [collected[k].cpu() for k in range(4)]; theta = None
        result = []
        for level, f in enumerate(fs):
            f = F.adaptive_avg_pool2d(f, (min(32,f.shape[-2]),min(32,f.shape[-1])))
            if theta is not None:
                f = F.grid_sample(f, source.inverse_feature_grid(theta,level,f.shape[-1]),mode='bilinear',align_corners=False,padding_mode='border')
            assert torch.isfinite(f).all(); result.append(f[0].numpy())
        return result, float(((logits > 0) == message).mean()), None if theta is None else theta.tolist()
    # Actual execution check against the real-data representation used by the baseline.
    real = torch.load(real_source(method), weights_only=True)
    meta = {r['image_name']:r for r in json.loads((ROOT / 'data/train/metadata.json').read_text())}
    checks = []
    for i in (0,84,169):
        name = real['names'][i]
        image = np.asarray(Image.open(ROOT / 'data/train/images' / f'{name}.png').convert('RGB'))
        bits = np.array(list(map(int,meta[name]['message'])))
        values, _, _ = features(image,bits)
        for k,(v,f) in enumerate(zip(values,real['features'])):
            expected = f[i].numpy(); error = float(abs(v-expected).max())
            normalized_rmse = float(np.sqrt(np.mean((v-expected)**2)) / max(np.sqrt(np.mean(expected**2)),1e-8))
            assert normalized_rmse < 1e-4, (method,i,k,error,normalized_rmse)
            checks.append(dict(image=name,level=k,max_error=error,normalized_rmse=normalized_rmse))
    shapes = [tuple(f.shape[1:]) for f in real['features']]; del real
    save(dest / 'feature_preflight.json',checks)
    arrays = [np.lib.format.open_memmap(dest/f'features_{k}.npy', mode='r+' if (dest/f'features_{k}.npy').exists() else 'w+', dtype=np.float32,shape=(4200,*shape)) for k,shape in enumerate(shapes)]
    mask_path = dest/'targets.npy'
    targets = np.lib.format.open_memmap(mask_path,mode='r+' if mask_path.exists() else 'w+',dtype=np.uint8,shape=(4200,128,128))
    archive = zipfile.ZipFile(OLD / 'results/sources/val2017.zip'); manifest=[]; seen=set(); started=time.perf_counter()
    for block in range(42):
        path = dest / 'chunks' / f'{block:03d}.json'
        if path.exists():
            rows=json.loads(path.read_text()); assert len(rows)==100
            manifest += rows; seen.update(r['cover_sha256'] for r in rows); continue
        if method=='stegastamp' and block<22:
            chunk=torch.load(OLD/f'results/shards/{block:03d}.pt',weights_only=True)
            ids=slice(block*100,(block+1)*100)
            for arr,value in zip(arrays,chunk['features']): arr[ids]=value.numpy()
            targets[ids]=chunk['target'].numpy()
            rows=[dict(r,pool_index=block*100+j,source='reused exact prior synthetic sample') for j,r in enumerate(old_manifest[ids])]
            for arr in arrays+[targets]:arr.flush()
            save(path,rows);manifest+=rows;seen.update(r['cover_sha256'] for r in rows)
            print(method,'REUSED',len(manifest),flush=True);continue
        rows=[]
        for index in range(block*100,(block+1)*100):
            record=records[order[index]]; raw=archive.read('val2017/'+record['file_name'])
            rng=np.random.default_rng(20261121+index)
            cover,region,shape,crop=source.cover_and_mask(raw,annotations.get(record['id'],[]),rng,['rectangle','ellipse','object'][index%3])
            cover_sha=hashlib.sha256(cover.tobytes()).hexdigest()
            distance=int(np.count_nonzero(real_dhash!=source.dhash(cover),axis=1).min())
            assert cover_sha not in real_hashes and cover_sha not in seen and distance>12, 'Unexpected source overlap: stop before changing pool.'
            seen.add(cover_sha)
            if index<2200: assert cover_sha==old_manifest[index]['cover_sha256']
            bits=rng.integers(0,2,wm.nbits,dtype=np.uint8)
            mode='clean' if index%20==0 else 'full' if index%20==1 else 'patch' if index%2 else 'restore'
            rectangle=None
            if mode=='clean': image=cover.copy(); mask=np.zeros((512,512),bool)
            elif mode in ('full','restore'):
                marked=wm.embed(cover,bits); mask=np.ones((512,512),bool) if mode=='full' else ~region
                image=np.where(mask[:,:,None],marked,cover)
            else:
                rectangle=bbox(region,2 if method=='trustmark' else None); x0,y0,x1,y1=rectangle
                marked=cover.copy(); marked[y0:y1,x0:x1]=wm.embed(cover[y0:y1,x0:x1],bits)
                mask=region;image=np.where(mask[:,:,None],marked,cover)
            assert np.array_equal(image[~mask],cover[~mask])
            values,accuracy,theta=features(image,bits)
            gt=source.downsample_mask(mask.astype(np.uint8))
            for arr,v in zip(arrays,values):arr[index]=v
            targets[index]=gt
            mse=float(np.mean((image.astype(float)-cover.astype(float))**2))
            row=dict(image_id=f'{method}_{index:04d}',pool_index=index,coco_id=record['id'],source_filename=record['file_name'],
                     cover_sha256=cover_sha,image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),min_real_dhash=distance,
                     crop=crop,embedding_bbox=rectangle,mode=mode,shape=shape,message=''.join(map(str,bits.tolist())),
                     area=float(gt.mean()),bit_accuracy=accuracy,psnr=None if mse==0 else 10*np.log10(255**2/mse),theta=theta,
                     split='validation' if 2000<=index<2200 else 'train',license=record['license'],flickr_url=record.get('flickr_url'))
            rows.append(row)
            if index<20 or 2200<=index<2210:
                Image.fromarray(image).save(dest/'examples'/f'{index:04d}_image.png')
                Image.fromarray(mask.astype(np.uint8)*255).save(dest/'examples'/f'{index:04d}_mask.png')
        for arr in arrays+[targets]:arr.flush()
        save(path,rows);manifest+=rows
        print(method,'GENERATED',len(manifest),'/4200',round(time.perf_counter()-started,1),'sec',flush=True)
    assert len(manifest)==len({r['coco_id'] for r in manifest})==len(seen)==4200
    save(dest/'manifest.json',manifest)
    save(dest/'complete.json',dict(method=method,count=4200,train_n=4000,holdout_n=200,seconds=time.perf_counter()-started,
                                 config_sha256=sha(dest/'config.json'),manifest_sha256=sha(dest/'manifest.json'),
                                 min_real_dhash=min(r['min_real_dhash'] for r in manifest),shapes=[list(s) for s in shapes]))
    assert sha(OLD/'results/full_train/submission.pt')==config['full_train_preserved']
    for h in handles:h.remove()
    print(method,'COMPLETE',flush=True)

if __name__=='__main__':main()
