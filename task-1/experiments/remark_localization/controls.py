"""Clean-photo, matched embedding, oracle-crop and spatial-alignment controls on GPU."""
import argparse
import io
import time
import zipfile
import cv2
import pandas as pd
from PIL import ImageOps
from scipy.stats import spearmanr
from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    worker = Remover(args.device)
    folder = OUT / 'controls'
    folder.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'experiments/synthetic_pretrain_2k/results/sources/val2017.zip'
    assert archive.exists()
    clean_rows, pair_rows, manifests, geometry = [], [], [], []
    rng = np.random.default_rng(20261121)
    with zipfile.ZipFile(archive) as z:
        files = sorted(n for n in z.namelist() if n.endswith('.jpg'))
        chosen = rng.choice(files, 24, replace=False)
        for index, filename in enumerate(chosen):
            raw = z.read(filename)
            original = Image.open(io.BytesIO(raw)).convert('RGB')
            cover = ImageOps.fit(original, (512, 512), method=Image.Resampling.LANCZOS)
            name = Path(filename).stem
            image = np.asarray(cover)
            cover.save(folder / f'{name}_clean.png')
            recovered, clean_maps, _ = worker.full(cover)
            np.savez_compressed(folder / f'{name}_clean_maps.npz', **clean_maps)
            magnitude = clean_maps['R1_s0']
            std, edge, hf = [down(v) for v in image_statistics(cover)]
            r = dict(image_id=name, residual_mean=float(magnitude.mean()), residual_p95=float(np.quantile(magnitude,.95)))
            for label, values in [('texture', std), ('edge', edge), ('highfrequency', hf)]:
                low, high = np.quantile(values, [.2,.8])
                r[label + '_rho'] = float(spearmanr(magnitude.ravel(), values.ravel()).statistic)
                r[label + '_low_residual'] = float(magnitude[values <= low].mean())
                r[label + '_high_residual'] = float(magnitude[values >= high].mean())
            clean_rows.append(r)
            manifests.append(dict(image_id=name, archive_member=filename, source_sha256=hashlib.sha256(raw).hexdigest(),
                                  processed_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(), original_size=list(original.size)))
            message = ''.join(str(int(v)) for v in rng.integers(0, 2, 100))
            expected = np.array([int(v) for v in message])
            watermarked, masks, details, oracle_inputs = [], [], [], []
            for size in (36, 89, 162, 256):
                x, y = [int(v) for v in rng.integers(8, 505 - size, 2)]
                crop = cover.crop((x,y,x+size,y+size))
                embedded = worker.tm.encode(crop, message, MODE='binary')
                wm = cover.copy(); wm.paste(embedded,(x,y))
                mask = np.zeros((512,512),bool); mask[y:y+size,x:x+size] = True
                watermarked.append(wm); masks.append(mask)
                details.append(dict(kind=f'patch{size}', x=x,y=y,size=size))
                oracle_inputs.append(embedded)
            full = worker.tm.encode(cover,message,MODE='binary')
            watermarked.append(full); masks.append(np.ones((512,512),bool)); details.append(dict(kind='full',x=0,y=0,size=512)); oracle_inputs.append(full)
            hole = full.copy(); hole.paste(cover.crop((192,192,320,320)),(192,192))
            mask = np.ones((512,512),bool);mask[192:320,192:320]=False
            watermarked.append(hole);masks.append(mask);details.append(dict(kind='clean_hole',x=192,y=192,size=128));oracle_inputs.append(hole)
            recovered_pairs = worker.patches(watermarked)
            oracle_removed = worker.patches(oracle_inputs)
            global_logits = worker.decode(watermarked + recovered_pairs)
            # Decoder receives different-size oracle crops separately, using competition resize.
            oracle_acc = []
            for crop, rec in zip(oracle_inputs, oracle_removed):
                logits = worker.decode([crop,rec]);oracle_acc.append((logits>0)==expected)
            clean_r1 = np.abs(image.astype(float)-np.asarray(recovered).astype(float)).mean(2)/255
            for j,(wm,rec,mask,info,oracle) in enumerate(zip(watermarked,recovered_pairs,masks,details,oracle_removed)):
                residual = np.abs(np.asarray(wm,np.float32)-np.asarray(rec,np.float32)).mean(2)/255
                before, after = (global_logits[[j,j+6]]>0)==expected
                inside = float(residual[mask].mean())
                outside = float(residual[~mask].mean()) if (~mask).any() else np.nan
                delta = residual-clean_r1
                row = dict(image_id=name,**info,gt_area=float(mask.mean()),residual_inside=inside,residual_outside=outside,
                           clean_residual_same_inside=float(clean_r1[mask].mean()),
                           residual_increase_inside=float(delta[mask].mean()),
                           residual_increase_outside=float(delta[~mask].mean()) if (~mask).any() else np.nan,
                           before_accuracy=float(before.mean()),after_accuracy=float(after.mean()),
                           oracle_before_accuracy=float(oracle_acc[j][0].mean()),oracle_after_accuracy=float(oracle_acc[j][1].mean()),
                           oracle_residual_mean=float(np.abs(np.asarray(oracle_inputs[j],np.float32)-np.asarray(oracle,np.float32)).mean()/255))
                pair_rows.append(row)
                np.savez_compressed(folder/f'{name}_{info["kind"]}.npz',score=down(residual),target=official.downsample_mask(mask.astype(np.uint8)).astype(bool),clean_score=down(clean_r1))
                if index < 4:
                    wm.save(folder/f'{name}_{info["kind"]}.png')
            # Image registration diagnostic, excluding API's feathered image margin.
            g1 = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)[16:-16,16:-16]
            g2 = cv2.cvtColor(np.asarray(recovered),cv2.COLOR_RGB2GRAY).astype(np.float32)[16:-16,16:-16]
            shift, response = cv2.phaseCorrelate(g1,g2)
            test = np.asarray(worker.patches([cover],batch_size=1)[0])
            geometry.append(dict(image_id=name,dx=float(shift[0]),dy=float(shift[1]),phase_response=float(response),
                                 batch_api_max_pixel_error=int(np.abs(test.astype(int)-np.asarray(recovered).astype(int)).max())))
            print('CONTROL',index+1,'/24',flush=True)
    pd.DataFrame(clean_rows).to_csv(folder/'clean_metrics.csv',index=False)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(folder/'matched_pairs.csv',index=False)
    pd.DataFrame(geometry).to_csv(folder/'alignment.csv',index=False)
    save(folder/'sources.json',dict(source='Official COCO val2017, photos without deliberately added TrustMark; unknown pre-existing provenance marks not independently certified',
                                  url='https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip',archive_sha256=sha(archive),images=manifests,
                                  generation='TrustMark0.8.0 Q,100raw bits,WM_STRENGTH1,WM_MERGEbilinear; local encode(crop) then paste same content; whole-image encode then clean-hole restoration'))
    means = pairs.groupby('kind').mean(numeric_only=True)
    means.to_csv(folder/'matched_summary.csv')
    save(folder/'summary.json',dict(clean_images=24,matched_images=len(pairs),device=args.device,gpu=torch.cuda.get_device_name(worker.device),
                                  clean_mean_residual=float(np.mean([r['residual_mean'] for r in clean_rows])),
                                  texture_rho=float(np.mean([r['texture_rho'] for r in clean_rows])),
                                  edge_rho=float(np.mean([r['edge_rho'] for r in clean_rows])),
                                  maximum_phase_shift=float(max(np.hypot(r['dx'],r['dy']) for r in geometry)),
                                  maximum_batch_api_pixel_error=max(r['batch_api_max_pixel_error'] for r in geometry),
                                  paired_summary=means.reset_index().to_dict('records')))
    print(means[['before_accuracy','after_accuracy','oracle_before_accuracy','oracle_after_accuracy','residual_increase_inside','residual_increase_outside']].to_string())


if __name__ == '__main__': main()
