"""Leakage-free inner threshold/map selection, macro localization and oracle diagnostics."""
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from common import *


def curve(score, gt, thresholds):
    # Match NumPy's float32 score >= Python-scalar threshold semantics at exact ties.
    thresholds = np.asarray(thresholds, dtype=score.dtype)
    order = np.argsort(score.ravel(), kind='stable')
    s, g = score.ravel()[order], gt.ravel()[order].astype(int)
    c = np.r_[0, np.cumsum(g)]
    k = np.searchsorted(s, thresholds, side='left')
    fn, tp = c[k], c[-1] - c[k]
    tn, fp = k - fn, len(g) - k - tp
    fg = np.divide(tp, tp + fp + fn, out=np.ones(len(k)), where=(tp + fp + fn) != 0)
    bg = np.divide(tn, tn + fp + fn, out=np.ones(len(k)), where=(tn + fp + fn) != 0)
    return (fg + bg) / 2


def boundary(g, p):
    a = g & ~ndi.binary_erosion(g, border_value=1)
    b = p & ~ndi.binary_erosion(p, border_value=1)
    if not a.any() or not b.any(): return 0. if a.any() or b.any() else np.nan
    pr = (ndi.distance_transform_edt(~a)[b] <= 2).mean()
    re = (ndi.distance_transform_edt(~b)[a] <= 2).mean()
    return 2 * pr * re / (pr + re) if pr + re else 0.


def one(score, gt, threshold):
    p = score >= threshold
    tp, fp, fn, tn = [int(a.sum()) for a in (p & gt, p & ~gt, ~p & gt, ~p & ~gt)]
    fg, bg = tp / (tp + fp + fn), tn / (tn + fp + fn)
    miou = official.miou_image(p, gt)
    assert abs(miou - (fg + bg) / 2) < 1e-12
    assert abs(curve(score, gt, [threshold])[0] - miou) < 1e-12
    count = int(gt.sum())
    order = np.argsort(-score.ravel(), kind='stable')
    topk = float(gt.ravel()[order[:count]].mean())
    cuts = np.quantile(score, np.linspace(0, 1, 101))
    cuts = np.r_[float(score.min()) - 1e-6, cuts, float(score.max()) + 1e-6]
    oracle = curve(score, gt, cuts)
    holes = ndi.binary_fill_holes(gt) & ~gt
    coreg, cores = gt[4:-4, 4:-4], score[4:-4, 4:-4]
    inside, outside = float(score[gt].mean()), float(score[~gt].mean())
    return dict(miou=float(miou), fg_iou=fg, clean_iou=bg, auc=float(roc_auc_score(gt.ravel(), score.ravel())),
                ap=float(average_precision_score(gt.ravel(), score.ravel())), topk_precision=topk,
                topk_lift=topk / float(gt.mean()), topk_cutoff_ties=int((score == score.ravel()[order[count-1]]).sum()),
                gt_area=float(gt.mean()), pred_area=float(p.mean()), inside=inside, outside=outside,
                separation=inside - outside, stronger_inside=inside > outside, boundary_f1=boundary(gt, p),
                hole_pixels=int(holes.sum()), hole_mean=float(score[holes].mean()) if holes.any() else np.nan,
                hole_fpr=float(p[holes].mean()) if holes.any() else np.nan,
                border_excluded_auc=float(roc_auc_score(coreg.ravel(), cores.ravel())) if coreg.any() and not coreg.all() else np.nan,
                border_excluded_ap=float(average_precision_score(coreg.ravel(), cores.ravel())) if coreg.any() and not coreg.all() else np.nan,
                oracle_miou=float(oracle.max()), oracle_threshold=float(cuts[oracle.argmax()]),
                threshold=float(threshold), fp=fp, fn=fn, tp=tp, tn=tn)


METRICS = ['miou','fg_iou','clean_iou','auc','ap','topk_precision','topk_lift','inside','outside','separation',
           'stronger_inside','boundary_f1','hole_mean','hole_fpr','oracle_miou','border_excluded_auc','border_excluded_ap','gt_area','pred_area']


def summarize(frame):
    rows = []
    selectors = [('all', np.ones(len(frame), bool)), ('le1pct', frame.gt_area <= .01),
                 ('1to5pct', (frame.gt_area > .01) & (frame.gt_area <= .05)),
                 ('5to20pct', (frame.gt_area > .05) & (frame.gt_area <= .2)),
                 ('20to50pct', (frame.gt_area > .2) & (frame.gt_area <= .5)), ('gt50pct', frame.gt_area > .5),
                 ('small_le5pct', frame.gt_area <= .05)]
    for label, select in selectors:
        for method, group in frame[select].groupby('method', sort=False):
            row = dict(method=method, stratum=label, n=len(group), hole_images=int((group.hole_pixels > 0).sum()))
            row.update({m: float(group[m].mean()) for m in METRICS})
            row['ap_lift'] = float((group.ap / group.gt_area).mean())
            row['hole_fpr_pooled'] = float(np.average(group.loc[group.hole_pixels > 0, 'hole_fpr'], weights=group.loc[group.hole_pixels > 0, 'hole_pixels'])) if (group.hole_pixels > 0).any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--include-local', action='store_true')
    args = ap.parse_args()
    names, gt, base, base_threshold, meta = real_data()
    reuse_full = args.include_local and (OUT / 'calibration.json').exists() and (OUT / 'per_image_metrics.csv').exists()
    previous_calibration = json.loads((OUT / 'calibration.json').read_text())['maps'] if reuse_full else []
    full_keys = set(np.load(OUT / 'full' / f'{names[0]}.npz').files)
    previous_calibration = [r for r in previous_calibration if r['method'] in full_keys]
    folders = (['local'] if reuse_full else ['full', 'local']) if args.include_local else ['full']
    maps = {}
    for folder in folders:
        files = [OUT / folder / f'{n}.npz' for n in names]
        assert all(p.exists() for p in files)
        keys = np.load(files[0]).files
        for key in keys:
            maps[key] = np.stack([np.load(p)[key] for p in files])
    splits = json.loads((BASE / 'folds.json').read_text())['trustmark']
    folds, calibrations, rows = [], list(previous_calibration), []
    if reuse_full:
        old = pd.read_csv(OUT / 'per_image_metrics.csv', float_precision='round_trip')
        old = old[old.method.isin(full_keys | {'baseline'})]
        assert old.groupby('method').size().eq(170).all() and len(old) == 170 * (len(full_keys)+1)
        rows = old.to_dict('records')
    # The existing baseline probabilities retain their own saved nested thresholds.
    for i, name in enumerate(names):
        if reuse_full: break
        rows.append(dict(image_id=name, method='baseline', fold=next(j+1 for j,s in enumerate(splits) if name in s['validation']),
                         **one(base[i], gt[i], float(base_threshold[i]))))
    for fold, split in enumerate(splits, 1):
        cal = np.array([names.index(n) for n in split['inner_validation']])
        val = np.array([names.index(n) for n in split['validation']])
        assert not set(cal) & set(val) and set(split['inner_validation']) <= set(split['train'])
        choices = [r for r in previous_calibration if r['fold'] == fold]
        for key, data in maps.items():
            candidates = np.unique(np.quantile(data[cal], np.linspace(0, 1, 101)))
            candidates = np.r_[float(data[cal].min())-1e-6, candidates, float(data[cal].max())+1e-6]
            scores = np.mean([curve(data[i], gt[i], candidates) for i in cal], axis=0)
            chosen = int(scores.argmax())
            threshold = float(candidates[chosen])
            choice = dict(fold=fold, method=key, threshold=threshold, calibration_miou=float(scores[chosen]),
                          calibration_names=split['inner_validation'], validation_names=split['validation'])
            choices.append(choice)
            calibrations.append(choice)
            for i in val:
                rows.append(dict(image_id=names[i], method=key, fold=fold, **one(data[i], gt[i], threshold)))
        families = {
            'B_full_selected': lambda k: k.startswith(('R1_', 'R2_', 'R3_', 'R4_')),
            'R5_selected': lambda k: k.startswith('R5_'),
            'generic_selected': lambda k: k.startswith('control_'),
        }
        if args.include_local: families['C_local_selected'] = lambda k: k.startswith('local')
        for label, rule in families.items():
            selected = max([r for r in choices if rule(r['method'])], key=lambda r: r['calibration_miou'])
            folds.append(dict(family=label, **selected))
            # Reuse metrics from the selected candidate; no outer-label-based choice.
            chosen_rows = [r for r in rows if r['fold'] == fold and r['method'] == selected['method']]
            rows.extend([dict(r, method=label, selected_map=selected['method']) for r in chosen_rows])
        print('EVALUATED fold', fold, 'maps', len(maps), flush=True)
    df = pd.DataFrame(rows)
    threshold_lookup = {(r['fold'],r['method']):r['threshold'] for r in calibrations}
    threshold_lookup.update({(r['fold'],r['family']):r['threshold'] for r in folds})
    baseline_lookup = dict(zip(names,base_threshold))
    df['threshold'] = [float(baseline_lookup[r.image_id]) if r.method=='baseline' else threshold_lookup[(r.fold,r.method)] for r in df.itertuples()]
    assert df.groupby('method').size().eq(170).all()
    df.to_csv(OUT / 'per_image_metrics.csv', index=False)
    summary = summarize(df)
    summary.to_csv(OUT / 'summary_by_area.csv', index=False)
    save(OUT / 'calibration.json', dict(protocol='existing inner_validation only; higher score predicts watermark', maps=calibrations, family_selection=folds))
    families = ['baseline','B_full_selected','R5_selected','generic_selected'] + (['C_local_selected'] if args.include_local else [])
    result = summary[(summary.stratum == 'all') & summary.method.isin(families)].to_dict('records')
    # Paired deltas are descriptive; candidate families selected inside each outer fold.
    reference = df[df.method == 'baseline'].set_index('image_id')
    comparisons = []
    for label in families[1:]:
        selected = df[df.method == label].set_index('image_id').loc[reference.index]
        differences = (selected.miou - reference.miou).to_numpy()
        rng = np.random.default_rng(20261121)
        ci = np.percentile(differences[rng.integers(0, 170, (2000, 170))].mean(1), [2.5,97.5])
        comparisons.append(dict(method=label, delta_miou=float(differences.mean()), ci=ci, improved_images=int((differences>0).sum())))
    save(OUT / 'summary.json', dict(status='local_complete' if args.include_local else 'full_complete', n_unique=170,
                                   families=result, comparisons=comparisons, map_count=len(set(r['method'] for r in calibrations)),
                                   all_metrics=summary.to_dict('records')))
    print(summary[(summary.stratum.isin(['all','small_le5pct'])) & summary.method.isin(families)][['method','stratum','n','miou','auc','ap','topk_precision','topk_lift']].to_string(index=False))


if __name__ == '__main__': main()
