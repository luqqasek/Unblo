"""Extract frozen full-image or fixed-window ReMark maps on GPU, no labels in inference."""
import argparse
import importlib.metadata
import json
import time
import trustmark

from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['full', 'local'], default='full')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=170)
    args = ap.parse_args()
    names, gt, base, thresholds, meta = real_data()
    worker = Remover(args.device)
    folder = OUT / args.mode
    folder.mkdir(parents=True, exist_ok=True)
    (OUT / 'recovered').mkdir(exist_ok=True)
    package = Path(trustmark.__file__).parent
    sources = [package / 'trustmark.py', package / 'denoise.py', package / 'KBNet/kbnet_s_arch.py',
               package / 'models/trustmark_rm_Q.ckpt', package / 'models/trustmark_rm_Q.yaml',
               package / 'models/encoder_Q.ckpt', package / 'models/decoder_Q.ckpt',
               BASE / 'trustmark_oof.npz', BASE / 'folds.json', ROOT / 'data/train/masks_128.npz']
    if args.mode == 'full':
        provenance = dict(version=importlib.metadata.version('trustmark'), torch=str(torch.__version__),
                          device=args.device, gpu=torch.cuda.get_device_name(worker.device),
                          cuda=torch.version.cuda, sources={str(p): sha(p) for p in sources},
                          constructor='Q, use_ECC=False, secret_len=100; 0.8.0 loads remover automatically; no loadRemover argument',
                          model_training=bool(worker.tm.removal.training), pretrained_weights_frozen=all(not p.requires_grad for p in worker.tm.removal.parameters()),
                          script_sha256=sha(__file__), common_sha256=sha(HERE / 'common.py'))
        save(OUT / 'provenance.json', provenance)
    start = time.perf_counter()
    for i, name in enumerate(names[:args.limit]):
        target = folder / f'{name}.npz'
        if target.exists(): continue
        image = Image.open(ROOT / 'data/train/images' / f'{name}.png').convert('RGB')
        if args.mode == 'full':
            recovered, maps, info = worker.full(image)
            bits = np.array([int(x) for x in meta[name]['message']])
            logits = worker.decode([image, recovered])
            decoded = logits > 0
            acc = (decoded == bits).mean(1)
            info.update(image_id=name, bit_accuracy_before=float(acc[0]), bit_accuracy_after=float(acc[1]),
                        changed_bits=int(np.count_nonzero(decoded[0] != decoded[1])),
                        significant_message_change=bool(np.count_nonzero(decoded[0] != decoded[1]) >= 10),
                        reliable_before=bool(acc[0] >= .75), accuracy_drop_ge10pp=bool(acc[0] - acc[1] >= .1))
            save(folder / f'{name}.json', info)
            recovered.save(OUT / 'recovered' / f'{name}.png')
        else:
            maps = worker.local(image)
        assert all(m.shape == (128, 128) and np.isfinite(m).all() for m in maps.values())
        np.savez_compressed(target, **maps)
        if i % 10 == 0 or i == args.limit - 1:
            print(f'{args.mode}: {i+1}/{args.limit} elapsed={time.perf_counter()-start:.1f}s GPU={args.device}', flush=True)
    print('DONE', args.mode, 'cached', len(list(folder.glob('*.npz'))), flush=True)


if __name__ == '__main__': main()
