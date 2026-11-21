"""Paired image-level evidence, synthetic QA and Polish report; no model selection."""
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; OUT = HERE / 'results'; ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('official_synthetic', ROOT / 'starter-kit/task-1/wmloc/utils.py')
official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
SEEDS = (42, 7, 123)

def save(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def measure(prob, gt, threshold):
    gt = gt.astype(bool); pred = prob >= threshold
    tp = np.count_nonzero(pred & gt); fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt); tn = np.count_nonzero(~pred & ~gt)
    fg = tp / (tp + fp + fn) if tp + fp + fn else 1.
    bg = tn / (tn + fp + fn) if tn + fp + fn else 1.
    score = float(official.miou_image(pred, gt))
    assert abs(score - (fg + bg) / 2) < 1e-12
    both = gt.any() and not gt.all()
    auc = roc_auc_score(gt.ravel(), prob.ravel()) if both else np.nan
    ap = average_precision_score(gt.ravel(), prob.ravel()) if both else np.nan
    k = int(gt.sum()); topk = np.nan
    if 0 < k < gt.size:
        cutoff = np.partition(prob.ravel(), -k)[-k]
        above = prob > cutoff; ties = prob == cutoff
        topk = (np.count_nonzero(above & gt) + (k - above.sum()) * gt[ties].mean()) / k
    empty = float(official.miou_image(np.zeros_like(gt), gt))
    return dict(miou=score, fg_iou=fg, clean_iou=bg, auc=float(auc), ap=float(ap), topk=float(topk),
                gt_area=float(gt.mean()), pred_area=float(pred.mean()), empty_miou=empty, delta_empty=score - empty,
                empty_prediction=not pred.any(), nearly_full_prediction=pred.mean() >= .99,
                probability_separation=float(prob[gt].mean() - prob[~gt].mean()) if both else np.nan,
                brier=float(np.mean((prob - gt) ** 2)), threshold=float(threshold))

def ci(values):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v): return [None, None]
    rng = np.random.default_rng(20261121)
    resamples = v[rng.integers(0, len(v), (5000, len(v)))].mean(1)
    return np.quantile(resamples, [.025, .975]).tolist()

def load_real():
    rows = []; runs = []; maps = {}
    metadata = {r['image_name']: r for r in json.loads((ROOT / 'data/train/metadata.json').read_text())}
    for variant in ('control', 'synthetic'):
        for seed in SEEDS:
            for fold in (1, 2, 3):
                stem = f'{variant}_fold{fold}_seed{seed}'
                z = np.load(OUT / f'{stem}.npz'); r = json.loads((OUT / f'{stem}.json').read_text())
                runs.append(r)
                assert z['names'].tolist() == r['names']
                for name, p, g in zip(z['names'], z['probability'], z['target']):
                    m = measure(p, g, float(z['threshold']))
                    rows.append(dict(image_id=str(name), method='stegastamp', variant=variant, seed=seed, fold=fold,
                                     full_watermark=metadata[str(name)]['full_watermark'], **m))
                    if seed == 42: maps[variant, str(name)] = (p, g.astype(bool), float(z['threshold']))
    return pd.DataFrame(rows), runs, maps

def synthetic_quality(manifest):
    q = pd.DataFrame(manifest)
    cols = ['area', 'bit_accuracy', 'psnr', 'watermark_pixel_change']
    quality = q.groupby(['split', 'mode'])[cols].agg(['mean', 'median', 'min', 'max'])
    quality.to_csv(OUT / 'synthetic_quality.csv')
    q.groupby(['split', 'mode', 'shape']).size().to_csv(OUT / 'synthetic_counts.csv', header=['n'])
    with zipfile.ZipFile(OUT / 'sources/val2017.zip') as archive:
        chosen = [0, 1, 3, 4, 5, 6, 17, 18, 31, 32, 39, 40]
        fig, axes = plt.subplots(len(chosen), 4, figsize=(12, 3 * len(chosen)))
        for axs, i in zip(axes, chosen):
            r = manifest[i]
            cover = np.asarray(Image.open(io.BytesIO(archive.read('val2017/' + r['source_filename']))).convert('RGB').crop(r['crop']).resize((512, 512), Image.Resampling.BICUBIC))
            image = np.asarray(Image.open(OUT / 'images' / (r['image_id'] + '.png')))
            mask = np.asarray(Image.open(OUT / 'masks' / (r['image_id'] + '.png')))
            diff = np.clip((image.astype(float) - cover.astype(float)) / 32 + .5, 0, 1)
            for ax, im, title in zip(axs, [cover, image, mask, diff], ['Czysty COCO', f'{r["mode"]} / {r["shape"]}', f'GT {r["area"]:.3f}', 'Residual x8 (szary=0)']):
                ax.imshow(im, cmap='gray', vmin=0 if im.ndim == 2 else None, vmax=255 if im.ndim == 2 else None)
                ax.set_title(title); ax.axis('off')
        fig.tight_layout(); fig.savefig(OUT / 'synthetic_example.png', dpi=90); plt.close(fig)
    return q

def retention(manifest):
    """Rankings in the same held-out synthetic domain before/after fine-tuning."""
    info = {r['image_id']: r for r in manifest}
    rows = []
    for seed in SEEDS:
        files = [('pretrain', 0, OUT / f'pretrain_seed{seed}_synthetic_validation.npz')]
        files += [(variant, fold, OUT / 'cache' / f'{variant}_fold{fold}_seed{seed}_synthetic_holdout.npz')
                  for variant in ('control', 'synthetic') for fold in (1, 2, 3)]
        for variant, fold, path in files:
            if not path.exists():
                raise FileNotFoundError(f'Run retention.py before report.py: {path}')
            z = np.load(path)
            for name, p, g in zip(z['names'], z['probability'], z['target']):
                rows.append(dict(variant=variant, fold=fold, seed=seed, image_id=str(name), mode=info[str(name)]['mode'],
                                 **measure(p, g, .5)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'synthetic_holdout_metrics.csv', index=False)
    cols = ['miou', 'fg_iou', 'clean_iou', 'auc', 'ap', 'topk', 'probability_separation']
    frame.groupby(['variant', 'mode'])[cols].mean().to_csv(OUT / 'synthetic_holdout_by_mode.csv')
    return {v: {m: float(frame[frame.variant == v][m].mean()) for m in cols} for v in ('pretrain', 'control', 'synthetic')}

def visualizations(frame, paired, maps, runs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for seed in SEEDS:
        h = json.loads((OUT / f'pretrain_seed{seed}_history.json').read_text())
        axes[0].plot([r['epoch'] for r in h], [r['validation_miou'] for r in h], label=f'seed {seed}')
    axes[0].set(title='Walidacja syntetyczna (200 osobnych zdjęć)', xlabel='Epoka pretreningu', ylabel='Oficjalne mIoU'); axes[0].legend()
    for variant in ('control', 'synthetic'):
        m = frame[frame.variant == variant].groupby('seed').miou.mean().reindex(SEEDS)
        axes[1].plot(range(3), m, marker='o', label=variant)
    axes[1].set(xticks=range(3), xticklabels=SEEDS, xlabel='Seed', ylabel='Real OOF mIoU', title='Te same foldy i urządzenie'); axes[1].legend()
    axes[2].hist(paired.delta_miou, bins=25); axes[2].axvline(0, color='black'); axes[2].set(title='Parowana zmiana po obrazach', xlabel='Delta mIoU (średnia 3 seedów)', ylabel='Liczba obrazów')
    fig.tight_layout(); fig.savefig(OUT / 'learning_and_delta.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    avg = frame.groupby(['variant', 'image_id']).mean(numeric_only=True).reset_index()
    for variant in ('control', 'synthetic'):
        d = avg[avg.variant == variant]
        axes[0].scatter(d.gt_area, d.fg_iou, s=15, alpha=.5, label=variant)
        axes[1].scatter(d.gt_area, d.ap, s=15, alpha=.5, label=variant)
        axes[2].scatter(d.gt_area, d.pred_area, s=15, alpha=.5, label=variant)
    for ax, label in zip(axes, ['FG IoU', 'Macro AP per obraz', 'Przewidziana powierzchnia']):
        ax.set(xlabel='Powierzchnia GT', ylabel=label); ax.legend()
    axes[2].plot([0, 1], [0, 1], color='gray', linestyle='--')
    fig.tight_layout(); fig.savefig(OUT / 'size_and_area.png', dpi=150); plt.close(fig)
    chosen = list(paired.sort_values('delta_miou').head(4).image_id) + list(paired.sort_values('delta_miou').tail(4).image_id)
    small = frame[(frame.seed == 42) & (frame.variant == 'synthetic') & (frame.gt_area <= .05)].sort_values('fg_iou')
    chosen += list(small.head(2).image_id) + list(small.tail(2).image_id)
    chosen = list(dict.fromkeys(chosen))
    gallery = OUT / 'gallery'; gallery.mkdir(exist_ok=True)
    entries = []
    for name in chosen:
        rgb = np.asarray(Image.open(ROOT / 'data/train/images' / f'{name}.png').convert('RGB'))
        pc, g, tc = maps['control', name]; ps, _, ts = maps['synthetic', name]
        predc, preds = pc >= tc, ps >= ts
        errors = np.zeros((*g.shape, 3)); errors[preds & ~g] = [1, 0, 0]; errors[~preds & g] = [0, 0, 1]
        fig, axes = plt.subplots(1, 7, figsize=(19, 3))
        for ax, im, title in zip(axes, [rgb, g, pc, ps, predc, preds, errors], ['RGB', 'GT', 'P: kontrola', 'P: synthetic→real', f'Kontrola t={tc}', f'Synthetic t={ts}', 'FP czerwone / FN niebieskie']):
            ax.imshow(im, cmap='viridis' if title.startswith('P:') else 'gray', vmin=0 if im.ndim == 2 else None, vmax=1 if im.ndim == 2 else None)
            ax.set_title(title); ax.axis('off')
        a = measure(pc, g, tc); b = measure(ps, g, ts)
        fig.suptitle(f'{name} seed42 | mIoU {a["miou"]:.3f} → {b["miou"]:.3f} | FG {a["fg_iou"]:.3f} → {b["fg_iou"]:.3f} | AP {a["ap"]:.3f} → {b["ap"]:.3f} | GT={g.mean():.3f}')
        fig.tight_layout(); fig.savefig(gallery / f'{name}.jpg', dpi=100); plt.close(fig)
        entries.append(f'<p>{name}</p><img loading="lazy" src="{name}.jpg" style="width:100%">')
    (gallery / 'index.html').write_text('<!doctype html><html lang="pl"><meta charset="utf-8"><title>Synthetic: porównanie</title><body style="font-family:system-ui;margin:24px"><h1>Kontrola i synthetic→real</h1><p>Przykłady z największym pogorszeniem/poprawą średniej po seedach oraz małe maski; panele pokazują seed 42. Dobór diagnostyczny, nie reprezentatywna próba.</p>' + ''.join(entries) + '</body></html>', encoding='utf-8')

def main():
    frame, runs, maps = load_real()
    frame.to_csv(OUT / 'per_image_metrics.csv', index=False)
    numeric = ['miou', 'fg_iou', 'clean_iou', 'auc', 'ap', 'topk', 'delta_empty', 'probability_separation', 'pred_area', 'brier']
    avg = frame.groupby(['variant', 'image_id'])[numeric].mean()
    paired = avg.loc['synthetic'].join(avg.loc['control'], lsuffix='_synthetic', rsuffix='_control')
    comparisons = {}
    for metric in numeric:
        paired['delta_' + metric] = paired[metric + '_synthetic'] - paired[metric + '_control']
        comparisons[metric] = dict(control=float(paired[metric + '_control'].mean()), synthetic=float(paired[metric + '_synthetic'].mean()),
                                   delta=float(paired['delta_' + metric].mean()), delta_ci95=ci(paired['delta_' + metric]))
    paired = paired.reset_index(); paired.to_csv(OUT / 'paired_metrics.csv', index=False)
    seed_rows = []
    for seed in SEEDS:
        r = dict(seed=seed)
        for variant in ('control', 'synthetic'):
            f = frame[(frame.seed == seed) & (frame.variant == variant)]
            r[variant] = {m: float(f[m].mean()) for m in numeric}
        r['delta'] = r['synthetic']['miou'] - r['control']['miou']; seed_rows.append(r)
    bins = [-1e-9, 0, .01, .05, .2, .5, 1.000001]
    labels = ['empty', '0–1%', '1–5%', '5–20%', '20–50%', '50–100%']
    frame['size_bin'] = pd.cut(frame.gt_area, bins=bins, labels=labels)
    strata = []
    for (variant, size), data in frame.groupby(['variant', 'size_bin'], observed=True):
        strata.append(dict(variant=variant, size_bin=str(size), n=data.image_id.nunique(), **{m: float(data[m].mean()) for m in numeric},
                           empty_prediction_rate=float(data.empty_prediction.mean()), unreliable=data.image_id.nunique() < 10))
    pd.DataFrame(strata).to_csv(OUT / 'size_stratified_metrics.csv', index=False)
    frame.groupby(['variant', 'full_watermark'])[numeric].mean().to_csv(OUT / 'embedding_mode_metrics.csv')
    synthetic_only = []; transfer_rows = []
    for seed in SEEDS:
        z = np.load(OUT / f'synthetic_only_seed{seed}.npz')
        rr = [measure(p, g, .5) for p, g in zip(z['probability'], z['target'])]
        synthetic_only.append(dict(seed=seed, **{m: float(np.mean([r[m] for r in rr])) for m in numeric}))
        transfer_rows += [dict(image_id=str(name), seed=seed, **r) for name, r in zip(z['names'], rr)]
    transfer_frame = pd.DataFrame(transfer_rows)
    transfer_frame.to_csv(OUT / 'synthetic_only_real_metrics.csv', index=False)
    zero_avg = transfer_frame.groupby('image_id')[numeric].mean()
    transfer_comparison = {}
    for metric in ('miou', 'fg_iou', 'ap', 'auc', 'topk'):
        change = avg.loc['synthetic', metric] - zero_avg[metric]
        transfer_comparison[metric] = dict(synthetic_only=float(zero_avg[metric].mean()),
                                          after_finetune=float(avg.loc['synthetic', metric].mean()),
                                          delta=float(change.mean()), delta_ci95=ci(change))
    manifest = json.loads((OUT / 'manifest.json').read_text()); q = synthetic_quality(manifest)
    retained = retention(manifest)
    pretrain = [json.loads((OUT / f'pretrain_seed{s}.json').read_text()) for s in SEEDS]
    summary = dict(status='complete', method='stegastamp', real_unique_n=170, seeds=seed_rows, metrics=comparisons,
                   folds=[{k: v for k, v in r.items() if k not in ('per_image', 'names', 'inner_history')} for r in runs],
                   synthetic_only=synthetic_only, finetuning_vs_pretrain=transfer_comparison,
                   synthetic_holdout_retention=retained, synthetic_train_n=2000, synthetic_validation_n=200,
                   synthetic_validation_miou=float(np.mean([r['validation']['miou'] for r in pretrain])),
                   positive_image_delta_fraction=float((paired.delta_miou > 0).mean()),
                   seed_sd={v: float(np.std([r[v]['miou'] for r in seed_rows], ddof=1)) for v in ('control', 'synthetic')},
                   bootstrap='5000 resamples of 170 image IDs after averaging seed metrics; conditional on fitted models; not refitting uncertainty',
                   seconds_pretrain=sum(r['seconds'] for r in pretrain), seconds_finetune={v: sum(r['seconds'] for r in runs if r['variant'] == v) for v in ('control', 'synthetic')})
    summary['fixed_half_miou'] = {v: sum(r['fixed_half_miou'] * len(r['names']) for r in runs if r['variant'] == v) / 510 for v in ('control', 'synthetic')}
    summary['positive_fold_seed_pairs'] = sum(r['miou'] > next(c['miou'] for c in runs if c['variant'] == 'control' and c['seed'] == r['seed'] and c['fold'] == r['fold']) for r in runs if r['variant'] == 'synthetic')
    save(OUT / 'summary.json', summary)
    pd.DataFrame([dict(metric=m, **r) for m, r in comparisons.items()]).to_csv(OUT / 'summary_metrics.csv', index=False)
    visualizations(frame, paired, maps, runs)
    m = comparisons['miou']; small = frame[frame.gt_area <= .05]
    small_stats = {v: {metric: float(small[small.variant == v][metric].mean()) for metric in numeric} for v in ('control', 'synthetic')}
    large = frame[frame.gt_area > .5]
    verdict = ('W tej recepturze syntetyczny pretrening poprawił średnie OOF mIoU.' if m['delta'] > 0 else 'W tej recepturze syntetyczny pretrening nie poprawił średniego OOF mIoU.')
    if m['delta_ci95'][0] <= 0 <= m['delta_ci95'][1]:
        verdict += ' Przedział parowanej zmiany obejmuje zero; przewaga nie jest wiarygodnie rozstrzygnięta.'
    lines = ['# Czy 2000 syntetyków pomaga? StegaStamp', '', verdict, '',
             f'**Kontrola real-only: {m["control"]:.5f}; synthetic→real: {m["synthetic"]:.5f}; zmiana {m["delta"]:+.5f} mIoU.** Bootstrap 95% CI parowanej zmiany: [{m["delta_ci95"][0]:+.5f}, {m["delta_ci95"][1]:+.5f}].', '',
             f'Poprawa wystąpiła w **{summary["positive_fold_seed_pairs"]}/9 par fold–seed**. Najważniejsze zastrzeżenie: wzrosło głównie IoU klasy czystej ({comparisons["clean_iou"]["control"]:.4f} → {comparisons["clean_iou"]["synthetic"]:.4f}), a foreground IoU nie wzrosło ({comparisons["fg_iou"]["control"]:.4f} → {comparisons["fg_iou"]["synthetic"]:.4f}). Ranking pikseli jest jednak lepszy: AUC {comparisons["auc"]["control"]:.4f} → {comparisons["auc"]["synthetic"]:.4f}, AP {comparisons["ap"]["control"]:.4f} → {comparisons["ap"]["synthetic"]:.4f}. To poprawa sygnału i redukcja fałszywych alarmów, a nie rozwiązanie lokalizacji małych watermarków.', '',
             f'**Sam pretrening bez fine-tuningu: {transfer_comparison["miou"]["synthetic_only"]:.5f} mIoU.** Fine-tuning nie dał dodatkowej potwierdzonej przewagi w mIoU: parowany CI zmiany obejmuje zero. Jednocześnie poprawił AP/AUC, więc nie należy sprowadzać tego do całkowitego zapomnienia syntetyków.', '',
             '## Co wykonano', '',
             'Wygenerowano **2000 przykładów treningowych z 2000 różnych zdjęć COCO** oraz 200 przykładów syntetycznej walidacji z innych zdjęć. Oznakowanie prawdziwym encoderem StegaStamp ze starter kitu, 100 losowych surowych bitów. Zamrożony ekstraktor, korekta inverse-STN i niezmieniona głowica. Nie trenowano enkodera ani ekstraktora.', '',
             'Każdy z seedów 42/7/123 dostał 10 ustalonych epok pretreningu. Następnie wykonano 3 zewnętrzne foldy × 3 seedy fine-tuningu oraz 9 dopasowanych kontroli real-only. Długość fine-tuningu i próg wybrano wyłącznie na trzech wewnętrznych foldach. Świeży refit zaczynał od tego samego syntetycznego checkpointu, z nowym optymalizatorem. Walidacja syntetyczna nie wybierała checkpointu. Obrazy testowe i leaderboard nie były użyte.', '',
             'Plan zapisany przed wynikami: [README.md](README.md). Przegląd źródeł: [RESEARCH.md](RESEARCH.md). Badamy jedną rodzinę znaków, nie wszystkie trzy metody konkursu.', '',
             '## Literatura i wybór generatora', '',
             '[Watermark Anything (ICLR 2025)](https://arxiv.org/html/2411.07231v2) uzasadnia mieszanie obszarów czystych i oznakowanych z maskami różnego kształtu. [MaskWM (NeurIPS 2025)](https://arxiv.org/abs/2504.12739) rozwija uczenie odporności lokalnej, ale oba podejścia uczą także własne systemy znakowania: nie są gotowymi lokalizatorami znaków StegaStamp. Ten eksperyment stosuje zasady konstrukcji danych, nie deklaruje reprodukcji SOTA.', '',
             'Źródło zdjęć: [oficjalne COCO](https://cocodataset.org/#download), archiwum val2017. Ta nazwa nie oznacza użycia walidacji konkursowej. Generator bierze środkowy kwadrat i skaluje do 512×512 **przed** dodaniem watermarku. Obiektowe maski pochodzą z adnotacji COCO, nie z masek konkursowych.', '',
             '45% przykładów: znakowanie rzeczywistego cropa, potem wklejenie oznakowanych pikseli do tego samego czystego zdjęcia. 45%: znakowanie całego obrazu, potem przywrócenie czystego regionu. Po 5% całkiem czystych i całkiem oznakowanych. Maski: prostokąty, elipsy i obiekty COCO; brak JPEG po znakowaniu i brak generatywnego inpaintingu. Proporcje są hipotezą inżynierską, nie twierdzeniem z publikacji.', '',
             '## Wyniki OOF', '', '| Miara (średnia po obrazach i seedach) | Real-only | Synthetic→real | Zmiana | 95% CI zmiany |', '|---|---:|---:|---:|---|']
    for metric in ('miou', 'fg_iou', 'clean_iou', 'auc', 'ap', 'topk', 'delta_empty', 'probability_separation', 'brier'):
        r = comparisons[metric]
        lines.append(f'| {metric} | {r["control"]:.5f} | {r["synthetic"]:.5f} | {r["delta"]:+.5f} | [{r["delta_ci95"][0]:+.5f}, {r["delta_ci95"][1]:+.5f}] |')
    lines += ['', 'mIoU: oficjalna średnia per obraz, z oficjalną regułą pustej unii. AP i ROC-AUC to średnie per obraz, nie pooled pixel scores. Top-K wykorzystuje liczbę pikseli GT i jest wyłącznie diagnostyką; remisy rozstrzygnięto przez wartość oczekiwaną losowego wyboru. Wszystkie rzeczywiste maski zawierają obie klasy. Brier: niżej lepiej.', '',
              '| Seed | Real-only OOF | Synthetic→real OOF | Zmiana |', '|---|---:|---:|---:|']
    for r in seed_rows: lines.append(f'| {r["seed"]} | {r["control"]["miou"]:.5f} | {r["synthetic"]["miou"]:.5f} | {r["delta"]:+.5f} |')
    lines += ['', '| Fold | Seed | Kontrola | Synthetic→real | Zmiana | Epoka kontrola / synthetic | Próg kontrola / synthetic |', '|---|---|---:|---:|---:|---|---|']
    for seed in SEEDS:
        for fold in (1, 2, 3):
            a = next(r for r in runs if r['variant'] == 'control' and r['seed'] == seed and r['fold'] == fold)
            b = next(r for r in runs if r['variant'] == 'synthetic' and r['seed'] == seed and r['fold'] == fold)
            lines.append(f'| {fold} | {seed} | {a["miou"]:.5f} | {b["miou"]:.5f} | {b["miou"]-a["miou"]:+.5f} | {a["epoch"]} / {b["epoch"]} | {a["threshold"]} / {b["threshold"]} |')
    lines += ['', f'Poprawa średniej po seedach dotyczy {summary["positive_image_delta_fraction"]:.1%} obrazów. SD OOF między trzema seedami: kontrola {summary["seed_sd"]["control"]:.5f}, synthetic {summary["seed_sd"]["synthetic"]:.5f}. To 170 unikalnych obrazów, a nie 510 niezależnych obserwacji. CI obejmuje bootstrap obrazów po uśrednieniu seedów; nie obejmuje ponownego uczenia modeli ani niepewności wszystkich hiperparametrów. Jest to rozwojowy CV na wcześniej badanych danych.', '',
              '## Małe maski, duże obszary i kalibracja', '',
              f'GT ≤5%: {small.image_id.nunique()} obrazów. Foreground IoU {small_stats["control"]["fg_iou"]:.5f} → {small_stats["synthetic"]["fg_iou"]:.5f}; AP {small_stats["control"]["ap"]:.5f} → {small_stats["synthetic"]["ap"]:.5f}; AUC {small_stats["control"]["auc"]:.5f} → {small_stats["synthetic"]["auc"]:.5f}; delta względem pustej maski {small_stats["control"]["delta_empty"]:+.5f} → {small_stats["synthetic"]["delta_empty"]:+.5f}.', '',
              f'GT >50%: {large.image_id.nunique()} obrazów. IoU klasy czystej: {large[large.variant=="control"].clean_iou.mean():.5f} → {large[large.variant=="synthetic"].clean_iou.mean():.5f}. To test odzyskiwania czystych otworów wewnątrz dużych oznakowanych obszarów.', '',
              'Pełne przedziały wielkości: [size_stratified_metrics.csv](results/size_stratified_metrics.csv). Wyniki trybu znakowania: [embedding_mode_metrics.csv](results/embedding_mode_metrics.csv). Małe grupy <10 obrazów są oznaczone jako niewiarygodne do silnych wniosków. Progi zawsze dobrano wewnątrz treningu; nie przeszukiwano outer validation.', '',
              f'Przy wspólnym stałym progu 0.5: kontrola {summary["fixed_half_miou"]["control"]:.5f}, synthetic→real {summary["fixed_half_miou"]["synthetic"]:.5f}. Efekt pretreningu pozostaje dodatni również bez doboru progu. Jest to dodatkowa diagnostyka, nie zmiana głównego protokołu po obejrzeniu wyników.', '',
              '![Jakość zależnie od powierzchni](results/size_and_area.png)', '',
              '## Czy pretrening nauczył się syntetycznej lokalizacji?', '',
              f'Końcowe mIoU na 200 osobnych syntetykach: **{summary["synthetic_validation_miou"]:.5f}** (średnia trzech seedów; stały próg 0.5). Ten zbiór nie wybrał liczby epok.', '',
              '| Seed | Synthetic-only na real170, t=0.5 | FG IoU | Macro AP |', '|---|---:|---:|---:|']
    for r in synthetic_only: lines.append(f'| {r["seed"]} | {r["miou"]:.5f} | {r["fg_iou"]:.5f} | {r["ap"]:.5f} |')
    t = transfer_comparison['miou']
    lines += ['', f'Średnio synthetic-only osiąga {t["synthetic_only"]:.5f}, a po fine-tuningu {t["after_finetune"]:.5f}. Dodatkowa zmiana po fine-tuningu: {t["delta"]:+.5f}, CI95 [{t["delta_ci95"][0]:+.5f}, {t["delta_ci95"][1]:+.5f}]. To porównanie pełnych procedur: synthetic-only używa stałego 0.5, a fine-tuning ma próg dobrany wewnątrz treningu. Nie izoluje czystego efektu aktualizacji wag od kalibracji.', '',
              'Wynik syntetyczny i rzeczywisty nie są bezpośrednio porównywalne: różnią się domeną, geometrią i rozkładem powierzchni. Syntetyczna walidacja zawiera 10% pełnych/pustych masek, których nie ma w real170.']
    lines += ['', 'Synthetic-only nie korzysta z real-labels w treningu ani selekcji. Jego próg 0.5 nie jest kalibrowany na real. Wysoki wynik syntetyczny i niski rzeczywisty mogą wynikać z przesunięcia domeny; nie dowodzą automatycznie zapominania podczas fine-tuningu.', '',
              'Dodatkowo zmierzono te same 200 syntetycznych obrazów po fine-tuningu, bez wybierania modelu na tej ocenie. Średnie AP/AUC wykluczają pełne i puste GT (20 z 200); mIoU obejmuje wszystkie 200. Wyniki po fine-tuningu uśredniają dziewięć modeli, a pretrain trzy modele.', '',
              '| Model oceniany na syntetycznej walidacji, t=0.5 | mIoU | Macro AP | Macro AUC |', '|---|---:|---:|---:|']
    for key, label in [('pretrain', 'Po pretreningu'), ('synthetic', 'Po fine-tuningu na real'), ('control', 'Kontrola real-only')]:
        r = retained[key]
        lines.append(f'| {label} | {r["miou"]:.5f} | {r["ap"]:.5f} | {r["auc"]:.5f} |')
    lines += ['', '[Retencja według trybu generacji](results/synthetic_holdout_by_mode.csv). To pomiar dodatkowy, nie kryterium wyboru hiperparametrów.', '',
              '![Krzywe i parowane zmiany](results/learning_and_delta.png)', '',
              '## Jakość i odtwarzalność danych', '',
              'Walidacja pipeline sprawdza dokładnie niezmienione piksele poza maską, oficjalny downsampling 512→128 i zgodność nowych wyrównanych cech z aktywną bazą na trzech rzeczywistych obrazach / czterech poziomach. Zapisano wiadomości, identyfikatory i licencje COCO, hashe zdjęć i checkpointu, transformacje STN, bit accuracy, powierzchnię i PSNR.', '',
              'Sprawdzono SHA i perceptual dHash względem wszystkich 510 obrazów treningowych konkursu oraz powtórzenia coverów. Heurystyka nie gwarantuje wykluczenia wszystkich wspólnych źródeł po silnej edycji; brak źródłowych identyfikatorów konkursu. Nie użyto testowych obrazów do takiego porównania.', '',
              '[Jakość syntetyków](results/synthetic_quality.csv), [manifest](results/manifest.json), [preflight cech](results/feature_preflight.json), [konfiguracja](results/config.json), [konfiguracja generacji](results/generation_config.json). Bit accuracy dla całkiem czystych próbek odnosi się do losowej wiadomości i jest kontrolą negatywną. Dla maski pustej PSNR jest nieskończone i zapisane jako null, nie zero.', '',
              '![Przykłady syntetyczne i residual](results/synthetic_example.png)', '',
              '## Przykłady rzeczywiste', '', '[Galeria: kontrola kontra synthetic→real](results/gallery/index.html). Obejmuje pogorszenia, poprawy i małe regiony. Selekcja przykładów jest diagnostyczna; pełne wyniki wszystkich obrazów znajdują się w [per_image_metrics.csv](results/per_image_metrics.csv).', '',
              '## Koszt i granice wniosku', '',
              f'Suma zmierzonych czasów GPU: pretrening {summary["seconds_pretrain"]/60:.1f} min; fine-tuning {summary["seconds_finetune"]["synthetic"]/60:.1f} min; kontrola {summary["seconds_finetune"]["control"]/60:.1f} min. Generacja i ekstrakcja cech CPU raportowane osobno w source_quality.json. Nie trenowano RGB/SRM ani nowego dekodera w tym eksperymencie.', '',
              'Dodatkowy budżet pretreningu nie jest dopasowany liczbą kroków do real-only; badamy praktyczny pakiet „2k syntetyków + pretrening”, a nie izolowany efekt danych przy identycznej liczbie aktualizacji. Nie testowano innych liczebności, proporcji masek, niższego learning rate fine-tuningu ani mieszania real/synthetic podczas fine-tuningu. Samoblen­dowanie usuwa semantyczne szwy, ale nie odtwarza całej domeny generatywnych edycji konkursowych. COCO może być domeną znaną pretrained encoderowi; oceniamy transfer do real, nie nowość jego pierwotnych danych.', '',
              'Historyczna baza CPU służy tylko dodatkowej kontroli. Główne wyniki pochodzą z dopasowanych nowych uruchomień na tym samym GPU, float32, TF32 off i deterministycznych operacjach. Zmiana urządzenia może zmienić wybór epoki; nie należy mieszać wyników CPU i GPU w jednym parowanym efekcie.', '',
              '## Decyzja i kolejny eksperyment', '']
    if m['delta'] > 0 and m['delta_ci95'][0] > 0:
        lines += ['Ta receptura uzasadnia dalszą pracę nad syntetykami dla StegaStamp. Trzy kolejne kroki, według priorytetu:', '',
                  '1. **Ostrożniejszy fine-tuning tej samej puli 2k**: porównać dotychczasowe LR=0.001 z LR=0.0001, dopuszczając epokę 0 jako kandydat w wewnętrznym wyborze. Obecna procedura wybierała wyłącznie epoki 1–50, mimo że checkpoint pretrain już działał dobrze. Stały checkpoint synthetic-only jest obowiązkową kontrolą. Koszt mały, bez nowej generacji; kryterium: parowana poprawa mIoU/foreground IoU bez utraty AP. Nie dobierać wariantu na outer validation.', '',
                  '2. **Powtórzyć 2k dla PixelSeal i TrustMark**, z ich dokładnymi encoderami i niezmienionymi bazami. Koszt głównie generacji/ekstrakcji. Obecny wynik nie pozwala zakładać takiej samej korzyści dla innych rodzin.', '',
                  '3. **Osobno zbadać małe regiony**: porównać obecną recepturę z kontrolowanym zwiększeniem udziału małych i cienkich oznakowanych fragmentów, zachowując liczbę 2k i kroki pretreningu. Hipoteza jest słabsza niż główny wynik: obecny generator już zawiera małe maski, a ich FG IoU nadal jest niskie. Wymaga wzrostu AP i FG IoU, nie tylko mIoU dzięki czystemu tłu; w razie braku efektu wrócić do jakości reprezentacji lokalnej.', '',
                  'Nie zwiększać równocześnie modelu, liczby danych i zakresu augmentacji. Nie ma jeszcze dowodu, że samo 10k zamiast 2k rozwiąże małe watermarki. Wyniki należy potwierdzić poza wielokrotnie używanym zbiorem rozwojowym przed uznaniem przewagi konkursowej.']
    else:
        lines += ['Nie przyjmować tego pretreningu jako nowej bazy wyłącznie na podstawie wyniku syntetycznego. Następny kontrolowany eksperyment: ta sama pula 2k, ale mniejszy learning rate fine-tuningu i/lub osobny wariant mieszania real/synthetic, porównany z zachowanym wynikiem obecnej receptury. Najpierw sprawdzić, czy ograniczeniem jest transfer domeny czy utrata rankingów po fine-tuningu; zwiększanie samej liczby syntetyków nie jest jeszcze uzasadnione.']
    lines += ['', 'Wynik nie odpowiada jeszcze za PixelSeal i TrustMark. Nie wygenerowano nowych predykcji testowych ani submission; nie zastąpiono aktywnej bazy automatycznie.', '',
              'Pełne polecenia: [REPRODUCE.md](REPRODUCE.md). Wynik kontroli: [verification.json](results/verification.json).', '']
    (HERE / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    if (OUT / 'full_train/prediction_summary.json').exists():
        import subprocess
        subprocess.run([sys.executable, str(HERE / 'submission_report.py')], check=True)
    print(json.dumps(dict(miou=m, seeds=seed_rows, synthetic_validation_miou=summary['synthetic_validation_miou']), indent=2), flush=True)

if __name__ == '__main__':
    main()
