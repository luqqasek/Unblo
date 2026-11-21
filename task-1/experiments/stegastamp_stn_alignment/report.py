"""Paired OOF evaluation, all seeds and size strata, plus inspectable prediction panels."""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
BASE = HERE.parent / 'stegastamp_stable_selection/results'
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('official_scoring', ROOT / 'starter-kit/task-1/wmloc/utils.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
METRICS = ['miou', 'auc', 'ap', 'fg_iou', 'clean_iou', 'boundary_f1', 'pred_area', 'area_error']


def measure(p, g, threshold):
    pred = p >= threshold
    tp, fp, fn, tn = [int(m.sum()) for m in (pred & g, pred & ~g, ~pred & g, ~pred & ~g)]
    a = g & ~ndi.binary_erosion(g, border_value=1)
    b = pred & ~ndi.binary_erosion(pred, border_value=1)
    if a.any() and b.any():
        precision = (ndi.distance_transform_edt(~a)[b] <= 2).mean()
        recall = (ndi.distance_transform_edt(~b)[a] <= 2).mean()
        boundary = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    else:
        boundary = 0. if a.any() or b.any() else np.nan
    fg_iou = tp / (tp + fp + fn) if tp + fp + fn else 1.
    clean_iou = tn / (tn + fp + fn) if tn + fp + fn else 1.
    miou = official.miou_image(pred, g)
    assert abs(miou - (fg_iou + clean_iou) / 2) < 1e-12
    area = float(g.mean())
    assert 0 < area < 1, 'Current cohort is mixed-class; explicitly handle other cases if extended.'
    return dict(miou=float(miou), fg_iou=fg_iou, clean_iou=clean_iou,
                auc=float(roc_auc_score(g.ravel(), p.ravel())), ap=float(average_precision_score(g.ravel(), p.ravel())),
                boundary_f1=float(boundary), gt_area=area, pred_area=float(pred.mean()), area_error=float(abs(pred.mean() - area)),
                missed_half_watermark=tp / (tp + fn) < .5,
                missed_clean_large=bool(area > .5 and tn / (tn + fp) < .5),
                wrong_boundary=bool(boundary < .1), zero_overlap=tp == 0, empty=not pred.any(),
                false_positive_majority=bool(tp + fp > 0 and tp < fp),
                small_failure=bool(area <= .05 and fg_iou < .1))


def compare(df):
    a = df[df.variant == 'raw'].set_index(['seed', 'image_id'])
    b = df[df.variant == 'aligned'].set_index(['seed', 'image_id']).loc[a.index]
    assert len(a) == len(b) and a.index.is_unique
    return {k: dict(raw=float(a[k].mean()), aligned=float(b[k].mean()), delta=float((b[k] - a[k]).mean())) for k in METRICS}


def main():
    data = json.loads((OUT / 'metrics.json').read_text())
    assert data['status'] == 'complete' and len(data['runs']) == 9
    old = np.load(HERE.parent / 'frozen_extractor/results/cv3/stegastamp_oof.npz')
    names = old['names'].tolist()
    gt = {n: g.astype(bool) for n, g in zip(names, old['target'])}
    rows, pictures = [], {}
    for run in data['runs']:
        stem = f"stegastamp_fold{run['fold']}_seed{run['seed']}"
        for variant, folder in [('raw', BASE), ('aligned', OUT)]:
            z = np.load(folder / f'{stem}.npz')
            assert z['names'].tolist() == run['names']
            for name, p in zip(z['names'].tolist(), z['probability']):
                metrics = measure(p, gt[name], float(z['threshold']))
                rows.append(dict(image_id=name, variant=variant, fold=run['fold'], seed=run['seed'], threshold=float(z['threshold']), **metrics))
                if run['seed'] == 42:
                    pictures[variant, name] = p >= float(z['threshold'])
    df = pd.DataFrame(rows)
    assert len(df) == 1020 and df.image_id.nunique() == 170
    assert df.groupby(['variant', 'seed']).size().eq(170).all()
    df.to_csv(OUT / 'per_image_metrics.csv', index=False)
    seeds = [dict(seed=s, metrics=compare(df[df.seed == s])) for s in (42, 7, 123)]
    strata = []
    for label, select in [('small_le5pct', df.gt_area <= .05), ('mid_5to50pct', (df.gt_area > .05) & (df.gt_area <= .5)), ('large_gt50pct', df.gt_area > .5)]:
        part = df[select]
        strata.append(dict(stratum=label, n_unique=part.image_id.nunique(), metrics=compare(part)))
    paired = df.pivot(index=['image_id', 'seed'], columns='variant', values='miou')
    image_delta = (paired.aligned - paired.raw).groupby('image_id').mean().to_numpy()
    rng = np.random.default_rng(20261121)
    ci = np.percentile(image_delta[rng.integers(0, 170, (2000, 170))].mean(1), [2.5, 97.5]).tolist()
    counts = []
    issues = ['wrong_boundary', 'missed_clean_large', 'missed_half_watermark', 'false_positive_majority', 'zero_overlap', 'small_failure', 'empty']
    for seed in (42, 7, 123):
        for variant in ('raw', 'aligned'):
            part = df[(df.seed == seed) & (df.variant == variant)]
            counts.append(dict(seed=seed, variant=variant, n=170, **{k: int(part[k].sum()) for k in issues}))
    fold_comparisons = []
    for run in data['runs']:
        fold_comparisons.append(dict(fold=run['fold'], seed=run['seed'], epoch=run['epoch'], threshold=run['threshold'],
                                     metrics=compare(df[(df.seed == run['seed']) & (df.fold == run['fold'])])))
    summary = dict(status='complete', n_unique=170, seeds=seeds, metrics=compare(df), strata=strata, folds=fold_comparisons,
                   improved_fold_seed_pairs=sum(r['delta'] > 0 for r in data['runs']), total_fold_seed_pairs=9,
                   image_bootstrap_miou_delta_ci95=ci,
                   bootstrap_note='Paired image resampling after averaging seed deltas; conditional on trained models, reused development folds; not 510 independent images.',
                   failure_counts=counts)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    flat = [dict(stratum=s['stratum'], n_unique=s['n_unique'], metric=m, **v) for s in strata for m, v in s['metrics'].items()]
    pd.DataFrame(flat).to_csv(OUT / 'size_stratified_metrics.csv', index=False)
    pd.DataFrame(counts).to_csv(OUT / 'failure_counts.csv', index=False)
    # Fixed by the earlier audit, before seeing aligned outcomes: worst/median/best RAW mIoU in each size bin.
    examples = {
        'small': ['53934df0429a', '34db63f9db87', 'de11c61674be'],
        'mid': ['5b7127ca0a3a', '4571a5b76263', 'bd93c8feb303'],
        'large': ['d02e15fbddf3', 'bb4b0481f38e', '42ee90cc410d'],
    }
    example_rows = []
    for label, chosen in examples.items():
        fig, axes = plt.subplots(3, 6, figsize=(15, 8))
        for axs, name in zip(axes, chosen):
            g, a, b = gt[name], pictures['raw', name], pictures['aligned', name]
            scores = df[(df.image_id == name) & (df.seed == 42)].set_index('variant')
            example_rows.append(dict(image_id=name, size_group=label, raw=float(scores.loc['raw', 'miou']), aligned=float(scores.loc['aligned', 'miou']), delta=float(scores.loc['aligned', 'miou'] - scores.loc['raw', 'miou'])))
            def errors(p):
                rgb = np.zeros((128, 128, 3), dtype=np.uint8)
                rgb[p & ~g] = [255, 40, 40]
                rgb[~p & g] = [40, 90, 255]
                return rgb
            images = [Image.open(ROOT / 'data/train/images' / f'{name}.png'), g, a, b, errors(a), errors(b)]
            titles = [name, 'GT: watermark=1', f"Raw mIoU={scores.loc['raw', 'miou']:.3f}", f"Aligned mIoU={scores.loc['aligned', 'miou']:.3f}", 'Raw: FP red / FN blue', 'Aligned: FP red / FN blue']
            for ax, pic, title in zip(axs, images, titles):
                ax.imshow(pic, cmap='gray', vmin=0, vmax=1 if np.asarray(pic).ndim == 2 else 255)
                ax.set_title(title, fontsize=8)
                ax.axis('off')
        fig.suptitle(f'StegaStamp full Head, seed 42: {label} GT regions', fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f'stegastamp_{label}_examples.png', dpi=120)
        plt.close(fig)
    pd.DataFrame(example_rows).to_csv(OUT / 'visual_examples.csv', index=False)
    main_score = summary['metrics']['miou']
    improved_seeds = sum(s['metrics']['miou']['delta'] > 0 for s in seeds)
    seed42 = next(s for s in seeds if s['seed'] == 42)['metrics']['miou']
    lines = ['# Eksperyment 1: pełna głowica StegaStamp po korekcie STN', '',
             '**Status: aktywna baza do dalszej optymalizacji — decyzja użytkownika.** Przyjęto wariant z poprawioną geometrią STN mimo mieszanych wyników CV, ponieważ usuwa potwierdzony błąd współrzędnych. Kolejne warianty porównujemy z tym modelem na tych samych foldach i seedach. Obowiązujący protokół opisuje [indeks eksperymentów](../README.md).', '',
             f"**Wynik:** średnie OOF mIoU {main_score['raw']:.5f} → {main_score['aligned']:.5f} ({main_score['delta']:+.5f}). Poprawa w {improved_seeds}/3 seedach i {summary['improved_fold_seed_pairs']}/9 parach fold–seed. Dla operacyjnego seedu 42: {seed42['raw']:.5f} → {seed42['aligned']:.5f} ({seed42['delta']:+.5f}).", '',
             'Korekta została zaimplementowana i pełna głowica została ponownie wytrenowana. Wynik jest mieszany: średnia poprawia się dzięki seedowi 123, seed 42 pogarsza się, a seed 7 jest niemal bez zmian. Warunkowy przedział bootstrap dla średniej różnicy obejmuje zero. Najlepszym zaakceptowanym zgłoszeniem pozostaje #1202; wybór nowej bazy rozwojowej nie zmienia tego wyniku ani domyślnego seedu 42.', '',
             'Wytrenowano od zera 9 pełnych głowic na wyrównanych cechach: 3 identyczne zewnętrzne foldy × seedy 42/7/123. Bazą są zapisane modele stable_selection, użyte przez najlepszy wariant #1202. Wszystkie 9 bazowych checkpointów ponownie odtworzyło zapisane predykcje. Nie użyto korekty wiadomością z eksperymentu #1299.', '',
             'Zmienia się wyłącznie geometria czterech map cech przed obecną klasą Head. Zastosowano odwrotność rzeczywistego STN z konwencją samplera 398/399, środkami convolutions i binów adaptive pooling, bilinear oraz border padding. Nowy cache zgadza się z cache sondy liniowej z tolerancją 1e-7. Nie przekształcano gotowych predykcji i nie wczytywano starych wag do nowego treningu.', '',
             'Architektura, liczba parametrów, loss 0,5 BCE + 0,5 Dice, AdamW 0,001, weight decay 0,0001, batch 8, max 50 epok, patience 8 pozostają identyczne. Dobór epoki: pooled mIoU przy 0,5 z trzech stratyfikowanych wewnętrznych foldów. Próg: maksimum na tych samych wewnętrznych predykcjach w [0,3;0,4;0,5;0,6;0,7]. Następnie świeży trening na całym outer-train. Identyczny algorytm selekcji nie oznacza wymuszenia tej samej wybranej epoki i progu.', '',
             '## Wyniki OOF według seedu', '', '| Seed | Raw mIoU | Aligned mIoU | Zmiana | Raw AUC | Aligned AUC |', '|---|---:|---:|---:|---:|---:|']
    for s in seeds:
        m, a = s['metrics']['miou'], s['metrics']['auc']
        lines.append(f"| {s['seed']} | {m['raw']:.6f} | {m['aligned']:.6f} | {m['delta']:+.6f} | {a['raw']:.6f} | {a['aligned']:.6f} |")
    lines += ['', '## Agregacja wszystkich seedów', '', '| Metryka | Raw | Aligned | Zmiana |', '|---|---:|---:|---:|']
    for metric, v in summary['metrics'].items():
        lines.append(f"| {metric} | {v['raw']:.6f} | {v['aligned']:.6f} | {v['delta']:+.6f} |")
    lines += ['', f"Poprawa mIoU w **{summary['improved_fold_seed_pairs']}/9** parach fold–seed. OOF waży wszystkie obrazy równo (foldy mają 57/57/56 obrazów); seedy również równo. To 170 unikalnych obrazów, nie 510 niezależnych próbek. Warunkowy bootstrap parowanej różnicy mIoU po obrazach, po uśrednieniu seedów: [{ci[0]:+.6f}; {ci[1]:+.6f}].", '',
              '## Wszystkie foldy i wybór parametrów', '', '| Fold | Seed | Raw mIoU | Aligned mIoU | Delta | Epoki aligned | Próg aligned |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in data['runs']:
        lines.append(f"| {r['fold']} | {r['seed']} | {r['baseline']:.6f} | {r['miou']:.6f} | {r['delta']:+.6f} | {r['epoch']} | {r['threshold']} |")
    lines += ['', '## Małe, średnie i duże obszary', '', '| GT area | Unikalne obrazy | Raw FG IoU | Aligned FG IoU | Raw boundary F1 | Aligned boundary F1 |', '|---|---:|---:|---:|---:|---:|']
    for s in strata:
        f, b = s['metrics']['fg_iou'], s['metrics']['boundary_f1']
        lines.append(f"| {s['stratum']} | {s['n_unique']} | {f['raw']:.6f} | {f['aligned']:.6f} | {b['raw']:.6f} | {b['aligned']:.6f} |")
    lines += ['', 'Boundary F1: granica wewnętrzna, bez ramki obrazu, tolerancja 2 komórki siatki 128 (8 px obrazu). AUC i AP są średnimi per obraz, nie pooled po pikselach.', '',
              '## Przykłady przed i po', '', 'Dobór ustalony na podstawie poprzedniego audytu: najgorszy, środkowy i najlepszy raw mIoU w każdej grupie wielkości, seed42. Nie wybierano obrazów według poprawy nowego wariantu.', '']
    for label in examples:
        lines.append(f'![Porównanie {label}](results/stegastamp_{label}_examples.png)')
    lines += ['', 'Po obejrzeniu 9 przykładów: misie (53934df0429a) nadal dostają niemal pełną maskę, a czysty prostokąt przy Big Benie (bb4b0481f38e) nadal znika. Żółty autobus (bd93c8feb303) poprawia mIoU około 0,739 → 0,783 i traci dodatkową fałszywą wyspę. Sygnalizator (d02e15fbddf3) poprawia 0,156 → 0,281, lecz pozostaje daleki od GT. Tort (de11c61674be) pogarsza 0,534 → 0,430 przez większe fałszywe obszary, a zegar (42ee90cc410d) 0,617 → 0,466 przez zanik czystego wycięcia. Poprawa geometrii nie usuwa dominujących problemów lokalizacji.', '']
    lines += ['', '## Ograniczenia i artefakty', '',
              'To porównanie rozwojowe na wielokrotnie analizowanych foldach. OOF etykiety służą wyłącznie ocenie, a nie wyborowi epoki/progu; nie jest to nowy nietknięty test. Nie wybieramy najlepszego seedu. Interpolacja również wygładza cechy, więc bez osobnej kontroli wygładzania nie można przypisać całej różnicy wyłącznie geometrii. Nie oceniano testu konkursowego ani leaderboardu; poprzednie checkpointy i zgłoszenia zachowano.', '',
              '[Podsumowanie JSON](results/summary.json), [metryki per obraz](results/per_image_metrics.csv), [wyniki według wielkości](results/size_stratified_metrics.csv), [liczebności problemów](results/failure_counts.csv), [weryfikacja](results/verification.json).', '',
              'Odtworzenie z katalogu repozytorium:', '```powershell',
              'task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stn_alignment/run.py',
              'task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stn_alignment/report.py',
              'task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stn_alignment/verify.py',
              'task-1/.venv/Scripts/python.exe task-1/dashboard/build.py', '```']
    (HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(metrics=summary['metrics'], improved=summary['improved_fold_seed_pairs'], ci=ci), indent=2))


if __name__ == '__main__':
    main()
