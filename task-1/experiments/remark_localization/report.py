"""Measured decision report, >=20 reviewable real examples, controls and distributions."""
import html
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from common import *


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    assert summary['status'] == 'local_complete'
    df = pd.read_csv(OUT / 'per_image_metrics.csv', float_precision='round_trip')
    table = pd.read_csv(OUT / 'summary_by_area.csv')
    calibration = json.loads((OUT / 'calibration.json').read_text())
    control = json.loads((OUT / 'controls/summary.json').read_text())
    pairs = pd.read_csv(OUT / 'controls/matched_pairs.csv')
    paired_rankings=[]
    for _,pair in pairs.iterrows():
        name=str(int(pair.image_id)).zfill(12)
        z=np.load(OUT/'controls'/f'{name}_{pair.kind}.npz')
        g=z['target']
        if g.any() and not g.all():
            paired_rankings.append(dict(image_id=name,kind=pair.kind,
                clean_auc=float(roc_auc_score(g.ravel(),z['clean_score'].ravel())),
                embedded_auc=float(roc_auc_score(g.ravel(),z['score'].ravel())),
                paired_difference_auc=float(roc_auc_score(g.ravel(),(z['score']-z['clean_score']).ravel()))))
    pd.DataFrame(paired_rankings).to_csv(OUT/'controls/matched_ranking.csv',index=False)
    clean = pd.read_csv(OUT / 'controls/clean_metrics.csv')
    names, gt, base, baset, meta = real_data()
    phases = [json.loads((OUT/'full'/f'{n}.json').read_text()) for n in names]
    decoder = pd.DataFrame(phases)
    decoder.drop(columns=['feature_shapes']).to_csv(OUT / 'decoder_metrics.csv', index=False)
    selected = {r['method']:r for r in summary['families']}
    all_scores = table[table.stratum == 'all'].set_index('method')
    small_scores = table[table.stratum == 'small_le5pct'].set_index('method')
    residual_candidates = [m for m in all_scores.index if m.startswith(('R1_', 'R2_', 'R3_', 'R4_', 'local'))]
    pair_contrast = pairs[pairs.kind.str.startswith('patch')].residual_increase_inside - pairs[pairs.kind.str.startswith('patch')].residual_increase_outside
    gate_rows = []
    for method in residual_candidates:
        r, sm = all_scores.loc[method], small_scores.loc[method]
        gate_rows.append(dict(method=method, auc=float(r.auc), auc_minus_generic=float(r.auc-selected['generic_selected']['auc']),
                             small_topk_lift=float(sm.topk_lift), controlled_localized_increase=float(pair_contrast.mean()),
                             passed=bool(r.auc>=.60 and r.auc-selected['generic_selected']['auc']>=.02 and sm.topk_lift>=1.5 and pair_contrast.mean()>0)))
    passed = any(r['passed'] for r in gate_rows)
    save(OUT/'integration_gate.json',dict(passed=passed,criteria='AUC>=.60; AUC advantage>=.02 vs inner-selected generic; small Top-K lift>=1.5; matched residual increase inside>outside',
                                         variants=gate_rows, exploratory_multiple_comparisons=True))
    # Quantify new TP and new FP when selected ReMark predictions supplement the baseline.
    complementary = []
    for label in ('B_full_selected','C_local_selected'):
        for i,name in enumerate(names):
            row = df[(df.method==label)&(df.image_id==name)].iloc[0]
            choice = next(r for r in calibration['family_selection'] if r['family']==label and r['fold']==int(row.fold))
            folder = 'local' if label.startswith('C_') else 'full'
            s = np.load(OUT/folder/f'{name}.npz')[choice['method']]
            pred = s>=row.threshold
            baseline = base[i]>=baset[i]
            complementary.append(dict(method=label,image_id=name,added_tp=int((pred&~baseline&gt[i]).sum()),added_fp=int((pred&~baseline&~gt[i]).sum()),
                                      union_miou=float(official.miou_image(pred|baseline,gt[i])),
                                      delta_miou=float(row.miou-df[(df.method=='baseline')&(df.image_id==name)].miou.iloc[0])))
    comp = pd.DataFrame(complementary)
    comp.to_csv(OUT/'complementarity.csv',index=False)
    complementary_means=comp.groupby('method').mean(numeric_only=True)
    # False positives on deliberately unwatermarked source photos, using REAL inner-calibrated thresholds.
    clean_fpr = []
    for choice in calibration['family_selection']:
        if choice['family'] not in ('B_full_selected','R5_selected','generic_selected'): continue
        for n in clean.image_id.astype(str):
            # Preserve COCO's 12-digit filename when CSV dtype inference removes zeroes.
            n=n.zfill(12)
            maps=np.load(OUT/'controls'/f'{n}_clean_maps.npz')
            clean_fpr.append(dict(image_id=n,fold=choice['fold'],family=choice['family'],selected_map=choice['method'],
                                  fraction_pred_watermark=float((maps[choice['method']]>=choice['threshold']).mean())))
    pd.DataFrame(clean_fpr).to_csv(OUT/'controls/clean_false_positives.csv',index=False)
    # Deterministic visual sample spans every area bin plus holes, baseline failures and strongest ranking examples.
    raw=df[df.method=='B_full_selected'].set_index('image_id').loc[names]
    baseline_rows=df[df.method=='baseline'].set_index('image_id').loc[names]
    chosen=[]
    def add(indices):
        for i in indices:
            i=int(i)
            if i not in chosen: chosen.append(i)
    area=gt.mean((1,2))
    for lo,hi in [(0,.01),(.01,.05),(.05,.2),(.2,.5),(.5,1)]:
        ids=np.where((area>lo)&(area<=hi))[0]
        ids=ids[np.argsort(raw.auc.to_numpy()[ids])]
        if len(ids):add(ids[np.linspace(0,len(ids)-1,min(4,len(ids))).astype(int)])
    add(np.argsort(raw.hole_pixels.to_numpy())[-3:])
    add(np.argsort(baseline_rows.fg_iou.to_numpy())[:3])
    add(np.argsort(raw.auc.to_numpy())[-3:])
    texture_correlations=[]
    for name in names:
        with np.load(OUT/'full'/f'{name}.npz') as z:
            texture_correlations.append(float(spearmanr(z['R1_s0'].ravel(),z['control_edges_s0'].ravel()).statistic))
    add(np.argsort(texture_correlations)[-2:])
    example=OUT/'examples';example.mkdir(exist_ok=True)
    gallery=[]
    for i in chosen:
        name=names[i];g=gt[i]
        image=Image.open(ROOT/'data/train/images'/f'{name}.png').convert('RGB')
        rec=Image.open(OUT/'recovered'/f'{name}.png').convert('RGB')
        rgb=np.asarray(image,np.float32)/255;recovered=np.asarray(rec,np.float32)/255
        residual=rgb-recovered
        row=raw.loc[name]
        choice=next(r for r in calibration['family_selection'] if r['family']=='B_full_selected' and r['fold']==int(row.fold))
        s=np.load(OUT/'full'/f'{name}.npz')[choice['method']]
        p=s>=row.threshold;b=base[i]>=baset[i]
        errors=np.zeros((128,256,3),np.uint8)
        for k,pred in enumerate((p,b)):
            view=errors[:,128*k:128*(k+1)]
            view[pred&~g]=[255,30,30];view[~pred&g]=[30,100,255]
        fig,axes=plt.subplots(2,4,figsize=(12,6.7))
        pictures=[image,rec,np.clip(.5+residual*20,0,1),s,g,p,b,errors]
        titles=['RGB','ReMark reconstruction','Signed RGB difference x20',choice['method'], 'Ground truth',f'ReMark mIoU {row.miou:.3f}',f'Baseline mIoU {baseline_rows.loc[name,"miou"]:.3f}','Errors: ReMark | baseline; FP red / FN blue']
        for j,(ax,pic,title) in enumerate(zip(axes.ravel(),pictures,titles)):
            ax.imshow(pic,cmap='magma' if j==3 else 'gray',interpolation='nearest')
            ax.set_title(title,fontsize=8);ax.axis('off')
        fig.suptitle(f'TrustMark {name} | GT {area[i]:.2%} | ReMark AUC {row.auc:.3f}, AP {row.ap:.3f}',fontsize=10)
        fig.tight_layout();fig.savefig(example/f'{name}.jpg',dpi=120);plt.close(fig)
        gallery.append(dict(image_id=name,area=float(area[i]),auc=float(row.auc),ap=float(row.ap),file=f'{name}.jpg'))
    assert len(gallery)>=20
    save(example/'index.json',gallery)
    content='<!doctype html><html lang="pl"><meta charset="utf-8"><title>ReMark — przegląd obrazów</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:20px}img{width:100%;height:auto}article{margin-bottom:30px}a{color:#1769a2}</style><h1>ReMark: porównanie na rzeczywistych obrazach</h1><p>Dobór obejmuje wielkości masek, dziury, błędy baseline i najlepszy ranking residualu. Każda heatmapa ma własny zakres kolorów; nie porównuj jej jasności między obrazami.</p>'
    content+=''.join(f'<article><h2>{r["image_id"]} — GT {r["area"]:.1%}</h2><img loading="lazy" src="{r["file"]}" alt="RGB, rekonstrukcja, residual, GT i błędy {r["image_id"]}"></article>' for r in gallery)
    (example/'index.html').write_text(content+'</html>',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    r1=df[df.method=='R1_s0']
    axes[0].hist(r1.inside,bins=25,alpha=.65,label='GT watermark');axes[0].hist(r1.outside,bins=25,alpha=.65,label='GT clean')
    axes[0].axvline(control['clean_mean_residual'],color='black',ls='--',label='COCO clean mean')
    axes[0].set(xlabel='Mean absolute RGB residual, units [0,1]',ylabel='Images');axes[0].legend(fontsize=8)
    axes[1].scatter(r1.outside,r1.inside,c=r1.gt_area,cmap='viridis',s=15)
    lim=max(r1.outside.max(),r1.inside.max());axes[1].plot([0,lim],[0,lim],'k--');axes[1].set(xlabel='Mean outside GT',ylabel='Mean inside GT')
    plot=table[(table.stratum=='small_le5pct')&table.method.isin(['baseline','B_full_selected','C_local_selected','generic_selected'])]
    axes[2].bar(np.arange(len(plot)),plot.ap);axes[2].set_xticks(np.arange(len(plot)),['Baseline','Full','Generic','Local'][:len(plot)],rotation=20)
    axes[2].set(ylabel='Macro AP, GT <=5%');fig.tight_layout();fig.savefig(OUT/'trustmark_residual_examples.png',dpi=140);plt.close(fig)
    labels=['baseline','B_full_selected','C_local_selected','R5_selected','generic_selected']
    large=table[table.stratum=='gt50pct'].set_index('method')
    large_holes=df[(df.method=='B_full_selected')&(df.gt_area>.5)&(df.hole_pixels>0)]
    lines=['# ReMark jako sygnał lokalizacji TrustMark', '',
           '**Zakres:** 170 rzeczywistych obrazów TrustMark; zapisane 3-fold OOF i inner_validation. GPU RTX 3070, trustmark 0.8.0, Q, 100 bitów, bez ECC. Ekstrakcja i usuwanie na GPU, agregacja metryk na CPU. Wagi removera zamrożone. Użyto '+str(summary['map_count'])+' map/wariantów skal i wygładzenia.', '',
           '## Wynik porównania', '', '| Metoda | mIoU | FG IoU | Clean IoU | AUC macro | AP macro | Top-K oracle | Internal BF1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        r=all_scores.loc[label];lines.append(f'| {label} | {r.miou:.4f} | {r.fg_iou:.4f} | {r.clean_iou:.4f} | {r.auc:.4f} | {r.ap:.4f} | {r.topk_precision:.4f} | {r.boundary_f1:.4f} |')
    lines+=['','B: pełny residual; C: okna 128/256/384 i średnia skal; R5: RMS kanałów cech; generic: blur/resize/gradient. Mapa i smoothing wybierane wyłącznie na inner_validation maksymalizacją skalibrowanego mIoU. Osobno raportujemy AUC/AP/Top-K każdej mapy do oceny jakości przestrzennej. Wybór najlepszej mapy z tabeli OOF po obejrzeniu wyników byłby dodatkową selekcją rozwojową.','',
            f'Najlepszy pojedynczy wariant lokalny według OOF AUC, **eksploracyjnie wskazany po wynikach**, to local_mean_s3: AUC {all_scores.loc["local_mean_s3","auc"]:.4f}, AP {all_scores.loc["local_mean_s3","ap"]:.4f}, mIoU {all_scores.loc["local_mean_s3","miou"]:.4f}. Small Top-K lift wynosi {small_scores.loc["local_mean_s3","topk_lift"]:.3f}, blisko, lecz poniżej uprzednio ustalonego minimum1,5. Nie traktujemy tej arbitralnej granicy jako dowodu braku jakiejkolwiek informacji; całościowo wyniki AP, masek i kontroli nie uzasadniają dalszego treningu tej receptury.', '',
            '**Integracja D:** '+('bramka spełniona — wymaga osobnego porównania lekkiej głowicy z kontrolą o tej samej liczbie parametrów.' if passed else 'bramka użytecznej lokalizacji NIE została spełniona. Nie trenowano nowej głowicy; rekomendacja: zakończyć tę ścieżkę w przetestowanej postaci.'),'',
            '## Osiem pytań badawczych','',
            f'1. **Czy zmiany są silniejsze w watermarked?** Dla surowego R1: średnio {r1.inside.mean():.6f} wewnątrz i {r1.outside.mean():.6f} poza GT; silniejszy sygnał wewnątrz w {int(r1.stronger_inside.sum())}/170 obrazach. Dla czystych COCO średni residual wynosi {control["clean_mean_residual"]:.6f}.',
            f'2. **Korelacja z lokalizacją?** Pełny wybierany residual ma AUC {all_scores.loc["B_full_selected","auc"]:.4f}; cechy R5 {all_scores.loc["R5_selected","auc"]:.4f}. To sygnał słabszy od baseline AUC {all_scores.loc["baseline","auc"]:.4f}.',
            f'3. **Małe watermarki vs losowy ranking?** W 45 obrazach <=5%: AP full {small_scores.loc["B_full_selected","ap"]:.4f}, local {small_scores.loc["C_local_selected","ap"]:.4f}, generic {small_scores.loc["generic_selected","ap"]:.4f}. Średnie Top-K/prevalence: full {small_scores.loc["B_full_selected","topk_lift"]:.2f}, local {small_scores.loc["C_local_selected","topk_lift"]:.2f}; referencja losowa =1. Top-K używa prawdziwej powierzchni, więc jest oracle, nie predykcją wdrożeniową.',
            f'4. **Czyste otwory?** Przy GT>50% clean IoU: full {large.loc["B_full_selected","clean_iou"]:.4f}, local {large.loc["C_local_selected","clean_iou"]:.4f}, baseline {large.loc["baseline","clean_iou"]:.4f}. FPR w zamkniętych otworach: full {large.loc["B_full_selected","hole_fpr"]:.4f}, local {large.loc["C_local_selected","hole_fpr"]:.4f}. Na tych samych {len(large_holes)} obrazach z otworami średni score full w otworach to {large_holes.hole_mean.mean():.6f}, w FG {large_holes.inside.mean():.6f}. Obszary clean połączone z ramką nie są nazywane otworami.',
            f'5. **Lokalne przetwarzanie?** AUC full {all_scores.loc["B_full_selected","auc"]:.4f} → local {all_scores.loc["C_local_selected","auc"]:.4f}; AP {all_scores.loc["B_full_selected","ap"]:.4f} → {all_scores.loc["C_local_selected","ap"]:.4f}. Wynik dla małych obszarów podano osobno; oracle cropy nie są używane do oceny metody wdrożeniowej.',
            f'6. **Komplementarność?** Prosta suma logiczna masek baseline OR residual daje mIoU full {complementary_means.loc["B_full_selected","union_miou"]:.4f}, local {complementary_means.loc["C_local_selected","union_miou"]:.4f}, wobec baseline {all_scores.loc["baseline","miou"]:.4f}. To diagnostyka ustalonej reguły, nie nauczona integracja. Zapisano dodatkowe TP i FP per obraz; różne błędy nie wystarczają jako dowód użytecznej komplementarności.',
            f'7. **Specyficzność watermarku?** Kontrola matched-content wykazuje średni wzrost residualu wewnątrz minus poza lokalnym watermarkiem {pair_contrast.mean():.6f}. Usuwanie działa także na czystych obrazach; średni Spearman residual–tekstura {control["texture_rho"]:.3f}, residual–krawędzie {control["edge_rho"]:.3f}. Znak tych korelacji nie jest założony z góry. Porównanie z generic i parami nie uzasadnia traktowania całego residualu jako watermarku.',
            '8. **Czy integrować?** '+('Bramka wstępna spełniona; sama mapa nie zastępuje kontrolowanego eksperymentu integracji.' if passed else 'Nie w tej postaci: residual nie przechodzi ustalonej bramki jakości, a koszty lokalnego przetwarzania są znaczne.'),'',
            '## Powierzchnia i czyste otwory','', '| Metoda | GT area | n | AP | Top-K | Clean IoU | Hole FPR | Obrazy z dziurami |','|---|---|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        for bin_name in ['le1pct','1to5pct','5to20pct','gt50pct']:
            r=table[(table.method==label)&(table.stratum==bin_name)].iloc[0]
            lines.append(f'| {label} | {bin_name} | {int(r.n)} | {r.ap:.4f} | {r.topk_precision:.4f} | {r.clean_iou:.4f} | {r.hole_fpr:.4f} | {int(r.hole_images)} |')
    lines+=['','Hole FPR macro po obrazach z co najmniej jednym zamkniętym otworem; pooled FPR i średni score w otworach są w CSV. NaN oznacza brak takich otworów. BF1 wyłącza ramkę obrazu i toleruje 2 komórki siatki128.', '',
            '## Diagnostyka oracle i ramki obrazu', '',
            '| Metoda | Kalibrowane mIoU | Oracle próg per obraz — niedeployable | AUC bez 16px brzegu | AP bez brzegu |',
            '|---|---:|---:|---:|---:|']
    for label in labels:
        r=all_scores.loc[label]
        lines.append(f'| {label} | {r.miou:.4f} | {r.oracle_miou:.4f} | {r.border_excluded_auc:.4f} | {r.border_excluded_ap:.4f} |')
    lines+=['', 'Oracle wybiera próg z 101 kwantyli score plus pełna/pusta maska, osobno na każdym ocenianym obrazie z użyciem jego GT. To diagnostyka, a nie dozwolona procedura inferencji. AUC bez brzegu dotyczy tylko obrazów z obiema klasami w pozostawionym wnętrzu; undefined są pomijane, nie zastępowane zerem.', '',
            '## Usuwanie wiadomości i geometria','',
            f'Globalna trafność surowych bitów na rzeczywistych obrazach: {decoder.bit_accuracy_before.mean():.4f} → {decoder.bit_accuracy_after.mean():.4f}. Co najmniej 10/100 bitów zmieniło się na {int(decoder.significant_message_change.sum())}/170 obrazach. Wśród {int(decoder.reliable_before.sum())} z początkową trafnością >=0,75 spadek >=0,10 wystąpił na {int((decoder.reliable_before & decoder.accuracy_drop_ge10pp).sum())}. Dla małych watermarków globalny odczyt może być losowy już przed usuwaniem.', '',
            f'API daje RGB512→RGB512 uint8. Sieć dostaje PIL bilinear256, tensor[-1,1]; zwraca poprawkę dodawaną do oryginalnego obrazu, z clip i feather1%. Wersja0.8.0 nie przyjmuje loadRemover — zawsze ładuje remover. Batch ścieżka lokalna różni się od pojedynczego API o maks. {control["maximum_batch_api_pixel_error"]} poziom uint8 w 24 kontrolach; zapisano to jako efekt numeryczny GPU. Maks. przesunięcie fazowe na czystych kontrolach to {control["maximum_phase_shift"]:.4f}px; API nie ma transformacji przestrzennej poza resize poprawki.', '',
            'Osobno oceniono AUC/AP po wycięciu 16px ramki. Kontrola resize512→256→512 mierzy odpowiedzi przy krawędziach, a float residual oddziela kwantyzację. Nie interpretujemy oracle cropów jako metody lokalizacji.', '',
            '## Wizualizacje i artefakty','',
            f'[Galeria {len(gallery)} rzeczywistych obrazów — osiem paneli na obraz](results/examples/index.html). Dobór obejmuje wszystkie przedziały powierzchni, closed holes, porażki baseline i najlepsze rankingi residualu. Czerwony: FP, niebieski: FN; różnica RGB ×20 względem szarości0,5.', '',
            '![Rozkłady residualu](results/trustmark_residual_examples.png)', '',
            '[Per-image CSV](results/per_image_metrics.csv), [metryki według wielkości](results/summary_by_area.csv), [kalibracja](results/calibration.json), [model/provenance](results/provenance.json), [kontrole matched](results/controls/matched_pairs.csv), [bramka integracji](results/integration_gate.json), [weryfikacja](results/verification.json). Mapy i rekonstrukcje są w results/full, results/local, results/recovered. Hashe checkpointów i źródeł zachowano; brak konkursowego testu i submission.', '',
            '## Odtworzenie','', 'Środowisko CUDA `.venv-rgb-cuda`; uruchom z katalogu repo:', '```powershell',
            'task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/extract.py --mode full',
            'task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/controls.py',
            'task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/extract.py --mode local',
            'task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/evaluate.py --include-local',
            'task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/report.py',
            'task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/verify.py',
            'task-1/.venv/Scripts/python.exe task-1/dashboard/build.py', '```','',
            'Badanie rozwojowe na znanych foldach; 170 unikalnych obrazów. Wielość map zwiększa ryzyko wyboru po wynikach. Oracle thresholds per-image, kwantyle101 oraz Top-K są jawnie niedeployable. Wyniki bez uczonej głowicy nie wykluczają wszystkich możliwych sposobów użycia ReMark, lecz wystarczają do decyzji o tej recepturze.', '',
            'Źródła pierwotne: [Adobe TrustMark](https://github.com/adobe/trustmark), [publikacja ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html). Szczegóły implementacji i cytowania: [RESEARCH.md](RESEARCH.md).']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'report_status.json',dict(examples=len(gallery),integration_gate_passed=passed,decision='test fusion' if passed else 'abandon tested residual approach'))
    print('Report generated; examples',len(gallery),'integration gate',passed,flush=True)


if __name__=='__main__':main()
