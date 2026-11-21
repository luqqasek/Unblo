"""Actual measured report and deterministic visual case selection; no model fitting."""
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

def table(frame,columns):
    labels=list(columns)
    lines=['| '+' | '.join(labels)+' |','|'+'|'.join(['---']*len(labels))+'|']
    for _,r in frame.iterrows():
        values=[]
        for key in columns.values():
            v=r[key]; values.append(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v))
        lines.append('| '+' | '.join(values)+' |')
    return lines

def visuals(model,df,comp,ora,records,b,g):
    out=OUT/f'eval_{model}'; gallery=OUT/'gallery';gallery.mkdir(exist_ok=True)
    base=df[df.variant=='baseline'].set_index('image_id')
    direct=df[df.variant=='direct_fixed'].set_index('image_id')
    inv=df[df.variant=='inverse_fixed'].set_index('image_id')
    selected=df[df.variant=='selected'].set_index('image_id')
    runtime=pd.read_csv(out/'runtime.csv').set_index('image_id')
    index={r['image_id']:i for i,r in enumerate(records)}
    cases=[]
    def add(ids,category):
        for n in ids:
            if n not in [x[0] for x in cases]: cases.append((n,category))
    small=direct[direct.gt_area<=.05]
    add(small.sort_values('ap',ascending=False).head(3).index,'small_best_forensic_AP')
    add(small.sort_values('ap').head(2).index,'small_missed')
    large=selected[(selected.gt_area>.5)&(selected.hole_pixels>0)]
    add(large.sort_values('hole_recall',ascending=False).head(3).index,'large_clean_holes')
    add(comp.sort_values('delta_miou',ascending=False).head(3).image_id,'highest_deployable_delta')
    add(comp.sort_values('delta_miou').head(3).image_id,'worst_deployable_delta')
    confidence=runtime.confidence if runtime.confidence.notna().any() else direct.pred_area*0+np.nan
    bad=direct[direct.miou<.3].index
    add(confidence.loc[bad].sort_values(ascending=False).head(2).index,'confidence_wrong_or_unavailable')
    add((direct.miou-inv.miou).abs().sort_values().head(2).index,'ambiguous_orientation')
    add(ora.sort_values('fixed_gain_over_trivial',ascending=False).head(3).image_id,'oracle_geometric_success')
    entries=[]
    for n,category in cases:
        i=index[n]; z=np.load(OUT/model/f'{n}.npz'); p=z['probability']; gt=g[i]
        row=selected.loc[n]; cc=comp.set_index('image_id').loc[n]; predp=p if cc.orientation=='direct' else 1-p
        bp=b[i]>=records[i]['baseline_threshold']; fp=predp>=row.threshold
        rgb=np.asarray(Image.open(OUT/'inputs/rgb'/f'{n}.png'))
        bg=np.asarray(Image.fromarray(rgb).resize((128,128))).astype(float)/255
        def overlay(pred):
            im=bg*.45; im[pred&~gt]=[1,0,0];im[~pred&gt]=[0,.4,1]; return im
        panels=[rgb,gt,b[i],p,1-p,fp,overlay(bp),overlay(fp)]
        labels=['RGB','GT watermark','Baseline P',f'{model} F','1-F',f'{cc.orientation} t={row.threshold:.4g}','Baseline FP red / FN blue','Forensic FP red / FN blue']
        fig,axes=plt.subplots(1,len(panels),figsize=(22,3.2))
        for ax,im,title in zip(axes,panels,labels):
            ax.imshow(im,cmap='viridis' if title.endswith(' P') or title.endswith(' F') or title=='1-F' else 'gray',vmin=0 if im.ndim==2 else None,vmax=1 if im.ndim==2 else None)
            ax.set_title(title,fontsize=8);ax.axis('off')
        fig.suptitle(f'{category} | {n} {records[i]["family"]} fold{records[i]["fold"]} | area={gt.mean():.3f} | mIoU baseline {base.loc[n,"miou"]:.3f} forensic {row.miou:.3f}')
        fig.tight_layout();filename=f'{model}_{n}.jpg';fig.savefig(gallery/filename,dpi=110);plt.close(fig)
        entries.append(dict(model=model,image_id=n,category=category,file=filename))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for family in METHODS:
        ids=base[base.family==family].index
        axes[0].scatter(base.loc[ids,'miou'],selected.loc[ids,'miou'],s=10,alpha=.6,label=family)
        axes[1].scatter(base.loc[ids,'ap'],direct.loc[ids,'ap'],s=10,alpha=.6)
        axes[2].scatter(base.loc[ids,'gt_area'],(selected.miou-base.miou).loc[ids],s=10,alpha=.6)
    for ax in axes[:2]:ax.plot([0,1],[0,1],'k--',lw=.7)
    axes[0].set(xlabel='Baseline mIoU',ylabel='Forensic selected mIoU');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Baseline AP',ylabel='Forensic direct AP')
    axes[2].axhline(0,color='black',lw=.7);axes[2].set(xlabel='GT area',ylabel='Forensic - baseline mIoU')
    fig.suptitle(model);fig.tight_layout();fig.savefig(OUT/f'{model}_scatter_examples.png',dpi=150);plt.close(fig)
    return entries

def main():
    records,b,g,_,_=cohort(); models=[p.name[5:] for p in sorted(OUT.glob('eval_*')) if (p/'summary.json').exists()]
    all_summary=[]; all_folds=[]; all_cases=[]; gates=[]; complements=[]; oracle_summaries=[]
    lines=['# Pretrained image forensics: lokalizacja watermarków','',
           'Eksperyment na 510 oznaczonych obrazach treningowych, 170 na rodzinę. Oficjalna siatka 128×128; trzy istniejące outer-foldy. Nie użyto obrazów testowych ani leaderboardu. Nie generowano nowych danych i nie dostrajano modeli forensic.',
           '', '## Protokół i źródła', '',
           'Baza seed42: PixelSeal i TrustMark frozen_extractor/cv3; StegaStamp synthetic_pretrain_2k synthetic_fold{1,2,3}_seed42, z inverse-STN. Baza ta odpowiada recepturze najlepszego zaakceptowanego submission, ale wynik OOF nie jest wynikiem publicznego testu 0,5235. Historyczna średnia trzech seedów StegaStamp 0,50719 również nie jest wynikiem pojedynczego seedu42.',
           '', 'Sprawdzono dokładną zgodność 510 ID, rodzin, foldów i masek z data/train/masks_128.npz. Źródła i SHA256 OOF: [manifest inferencji](results/inference_trufor.json). Oficjalny kod, wagi, licencje i preprocessing: [PROVENANCE.md](PROVENANCE.md). Reprodukcja: [REPRODUCE.md](REPRODUCE.md).',
           '', 'Forensic jest zamrożony. Osobno dla każdej rodziny i outer-foldu progi direct i inverse dobieramy na istniejącym inner_validation (23 obrazy); wybór kierunku selected również korzysta wyłącznie z tego inner_validation. Żaden outer-validation GT nie wybiera progu ani kierunku wdrożeniowego. To porównanie rozwojowe na wielokrotnie oglądanych foldach, nie nowy niezależny test.',
           '', 'Zapisujemy float32 mapy 512×512, area-average do128×128, binarne maski i dopełnienia. Native_binary: oficjalny próg na512, następnie nearest do128. Kalibracja ma 101 kwantyli inner-map, dodatkowe progi0,05…0,95 oraz maski pełną/pustą. TruFor używa RGB/256 i wewnętrznej normalizacji; Mesorch RGB/255 z zewnętrzną normalizacją ImageNet; DetectiveSAM używa oficjalnego prepare_sample bez GT i oficjalnych wewnętrznych strumieni blur/noise. Nie są to nowe przykłady treningowe.',
           '', 'Metryki liczone per obraz, następnie uśredniane. AUC/AP pomijają obrazy z jedną klasą; liczebności zapisane w CSV. Top-K używa prawdziwej powierzchni GT i sprawiedliwie rozdziela remisy. BF1: tolerancja2 piksele siatki128 (8 RGB), zewnętrzna rama obrazu pominięta. Otwory: zamknięte składowe tła z binary_fill_holes; clean-hole recall to odzyskana część pikseli, component recall wymaga odzyskania≥50% składowej. To definicja topologiczna, nie etykieta typu edycji.',
           '', 'Oracle_orientation_fixed wybiera kierunek per obraz z GT przy progu0,5. Oracle_orientation_threshold dodatkowo sprawdza **każdą unikalną wartość** mapy, włącznie z pełną/pustą maską. Oba są tylko diagnostyką. Oracle_empty_full stanowi kontrolę dominującej klasy. Cztery przesunięcia toroidalne zachowują histogram, kształt i powierzchnię mapy, zmieniając położenie; ich oracle AUC porównujemy z nieprzesuniętą mapą.',
           '', '## Wyniki modeli']
    for model in models:
        out=OUT/f'eval_{model}'; summary=json.loads((out/'summary.json').read_text()); gate=summary['gate'];gates.append(gate)
        s=pd.read_csv(out/'summary_by_family_area.csv');s['model']=model;all_summary.append(s)
        df=pd.read_csv(out/'per_image_metrics.csv'); comp=pd.read_csv(out/'complementarity.csv');ora=pd.read_csv(out/'oracle_diagnostics.csv')
        folds=pd.read_csv(out/'per_fold.csv');folds['model']=model;all_folds.append(folds)
        all_cases.extend(visuals(model,df,comp,ora,records,b,g))
        lines+=['',f'### {model}','']
        t=s[(s.family=='all')&(s.stratum=='all')]
        lines+=table(t,{'Wariant':'variant','mIoU':'miou','FG IoU':'fg_iou','Clean IoU':'clean_iou','AUC':'auc','AP':'ap','BF1':'boundary_f1'})
        lines+=['',f'Inferencja: mediana {summary["runtime"]["median_seconds"]:.3f}s/obraz; maksimum allocated {summary["runtime"]["peak_allocated_mb"]:.1f}MiB. Czasy pełnych przebiegów Mesorch i DetectiveSAM obejmują okres współdzielenia RTX3070; nie są kontrolowanym rankingiem szybkości. Osobne pola reserved, czas per obraz i preprocessing SAM w cache.', '',
                f'Geometria: {gate["spatial_images"]}/510 obrazów, na których oracle przy0,5 przekracza oracle empty/full o>0,05. Z tego {gate["complementary_images"]} przekracza także bazę o>0,05; {gate["complementary_small_or_large"]} dotyczy ≤5% lub>50% GT. Średni oracle AUC {gate["oracle_auc"]:.4f}, po przesunięciu {gate["shift_oracle_auc"]:.4f}; różnica {gate["mean_shift_adjusted_oracle_auc"]:+.4f}, CI95 {gate["auc_shift_ci95"]}.', '',
                'Częstość kierunków oracle przy0,5: '+str(ora.fixed_orientation.value_counts().to_dict())+'. Przy optymalnym progu: '+str(ora.threshold_orientation.value_counts().to_dict())+'. Remisy przypisujemy direct.', '',
                f'Bramka feasibility: **{"GO" if gate["pass"] else "NO-GO"}**. Ustalona przed pierwszym pełnym wynikiem: ≥10 komplementarnych obrazów, ≥5 small/large, obecność we wszystkich3 foldach, przewaga oracle AUC nad przesunięciami>0,03. To praktyczne kryterium kontynuacji, nie test hipotezy zerowej ani dowód nieistnienia sygnału.', '',
                f'[Pełne metryki](results/eval_{model}/per_image_metrics.csv), [rodziny i powierzchnie](results/eval_{model}/summary_by_family_area.csv), [foldy](results/eval_{model}/per_fold.csv), [oracle](results/eval_{model}/oracle_diagnostics.csv), [kalibracja z ID](results/eval_{model}/calibration.json).', '',f'![Porównanie {model}](results/{model}_scatter_examples.png)']
        for family in ['all',*METHODS]:
            d=comp if family=='all' else comp[comp.family==family]
            oc=ora if family=='all' else ora[ora.family==family]
            cm=dict(model=model,family=family,n=len(d),error_correlation=float(d.error_correlation.mean()),
                    error_overlap_iou=float(d.error_overlap_iou.mean()),baseline_failures_improved=int(((d.baseline_miou<.4)&(d.delta_miou>.05)).sum()),
                    previously_good_damaged=int(((d.baseline_miou>=.5)&(d.delta_miou<-.05)).sum()),
                    any_improved=int((d.delta_miou>0).sum()),any_damaged=int((d.delta_miou<0).sum()),
                    oracle_best_system_gain=float((d.oracle_best_system_miou-d.baseline_miou).mean()),
                    delta_miou=float(d.delta_miou.mean()),ci95=ci(d.delta_miou))
            complements.append(cm)
            oracle_summaries.append(dict(model=model,family=family,direct_fixed_wins=int((oc.fixed_orientation=='direct_fixed').sum()),
                                         inverse_fixed_wins=int((oc.fixed_orientation=='inverse_fixed').sum()),
                                         exact_oracle_gain_over_trivial=float(oc.gain_over_trivial.mean()),
                                         fixed_gain_over_trivial=float(oc.fixed_gain_over_trivial.mean())))
    merged=pd.concat(all_summary,ignore_index=True);merged.to_csv(OUT/'summary_by_family_area.csv',index=False)
    foldframe=pd.concat(all_folds,ignore_index=True);foldframe.to_csv(OUT/'per_fold.csv',index=False)
    pd.DataFrame(complements).to_csv(OUT/'complementarity_summary.csv',index=False)
    pd.DataFrame(oracle_summaries).to_csv(OUT/'oracle_summary.csv',index=False)
    lines+=['','## Rodziny i najtrudniejsze obszary','']
    sel=merged[(merged.family!='all')&(merged.stratum=='all')&merged.variant.isin(['baseline','selected'])]
    lines+=table(sel,{'Model':'model','Rodzina':'family','Wariant':'variant','mIoU':'miou','AUC':'auc','AP':'ap','BF1':'boundary_f1'})
    for stratum,title in [('small_le5pct','Małe watermarki ≤5%'),('gt50pct','Duże watermarki >50%')]:
        lines+=['',f'### {title}','']
        t=merged[(merged.family=='all')&(merged.stratum==stratum)&merged.variant.isin(['baseline','direct_fixed','inverse_fixed','selected'])]
        cols={'Model':'model','Wariant':'variant','N':'n','FG IoU':'fg_iou','AP':'ap','Top-K oracle':'topk_precision','Miss rate':'complete_miss','FP area':'false_positive_area'} if stratum=='small_le5pct' else {'Model':'model','Wariant':'variant','N':'n','Clean IoU':'clean_iou','Internal BF1':'internal_boundary_f1','Hole recall':'hole_recall','Clean FPR':'clean_fpr'}
        lines+=table(t,cols)
    lines+=['','## Komplementarność, foldy i niepewność','',
            'Failure: baza mIoU<0,4; poprawa oznacza delta>0,05. Previously-good: baza≥0,5; uszkodzenie oznacza delta<−0,05. To opisowe progi diagnostyczne. Korelacja błędów jest Pearsonem masek błędu per obraz; brak wariancji daje NaN. Odmienność błędów sama nie oznacza przydatności.', '']
    lines+=table(pd.DataFrame(complements).query("family=='all'"),{'Model':'model','Korelacja błędów':'error_correlation','IoU błędów':'error_overlap_iou','Poprawione failures':'baseline_failures_improved','Uszkodzone dobre':'previously_good_damaged','Delta mIoU':'delta_miou'})
    for model in models:
        pair=json.loads((OUT/f'eval_{model}/paired_bootstrap.json').read_text())
        p=next(r for r in pair if r['family']=='all' and r['variant']=='selected')
        lines+=['',f'{model}: selected−baza {p["delta_miou"]:+.5f}, parowany bootstrap obrazów CI95 {p["ci95"]}. 5000 resampli, seed20261121; przedział warunkowy względem zapisanych modeli, nie obejmuje losowości treningu ani selekcji kierunku badawczego.']
    ff=foldframe[(foldframe.variant.isin(['baseline','selected']))]
    lines+=['']+table(ff,{'Model':'model','Rodzina':'family','Fold':'fold','Wariant':'variant','mIoU':'miou','AP':'ap'})
    passed=[x['model'] for x in gates if x['pass']]
    lines+=['','## Decyzja o fuzji','']
    lines += ['Bramkę przechodzą: '+', '.join(passed)+'. Wyniki nauczonej fuzji są raportowane osobno; samo przejście bramki nie dowodzi zysku.'] if passed else ['**NO-GO dla uczenia fuzji w tej recepturze.** Żaden z trzech modeli nie spełnia całej bramki użytecznej, komplementarnej geometrii. Zgodnie z planem nie trenowano E1/E2/E3 ani kontroli calibration-only. Nie użyto zwykłych globalnych OOF jako pozornie bezpiecznych danych treningowych fuzji: wymagałoby to inner cross-fitting, aby wykluczyć pośredni dostęp do outer-validation.']
    lines+=['','## Odpowiedzi na osiem pytań','',
            '1. **Czy pretrained manipulation models znajdują użyteczne regiony?** Zdarzają się pojedyncze trafienia geometryczne; ich liczbę ponad empty/full i bazę podają bramki powyżej. To nie dowód na użyteczność całego systemu.',
            '2. **Czy granice odpowiadają granicom watermarku?** Oceniamy BF1 i wewnętrzny BF1 w tabelach; galerie pokazują zarówno trafienia, jak i niepowiązane obiekty. Wysoki oracle mIoU bez przewagi nad pustą/pełną maską nie jest takim dowodem.',
            '3. **Direct czy inverse?** Nie ma bezpiecznej reguły „manipulacja=watermark”. Histogram zwycięstw oracle pokazuje rozbieżność, lecz jest zależny od GT i powierzchni. Metadata nie zawiera wiarygodnych etykiet wszystkich czterech typów edycji, więc nie przypisujemy im wyników na podstawie domysłów.',
            '4. **Czy kierunek można wybrać bez GT?** Przetestowano rodzinny wybór kierunku i progu na inner_validation. Wynik selected jest wdrożeniowy w sensie braku dostępu do outer GT, ale nie jest image-level gate. Możliwość skutecznego gate per obraz pozostaje nieudowodniona.',
            '5. **Czy połączenie poprawia prawdziwą lokalizację?** Oracle najlepszego systemu jest wyłącznie górnym diagnostycznym wskazaniem. Bez pozytywnej bramki nie uczymy połączenia; nie twierdzimy, że przetestowano wszystkie możliwe fuzje.',
            '6. **Który model jest najlepszy na rodzinie?** Ranking poniżej dotyczy selected i jest opisem po obejrzeniu OOF, nie nową regułą wyboru modelu do inferencji. Ranking AP może się różnić od mIoU.',
            '7. **Czy wzrost jest wiarygodny między foldami?** Foldy i parowane CI podano powyżej; brak nowej niezależnej walidacji oraz tylko seed42 dla porównania baz ograniczają uogólnienia.',
            '8. **Czy rozwijać kierunek?** '+('Wykonać minimalną, poprawnie cross-fitted próbę tylko modeli spełniających bramkę; decyzję końcową oprzeć na jej przewadze nad kontrolą.' if passed else 'Zakończyć tę ścieżkę jako priorytet dalszego rozwoju. Zachować mapy do diagnostyki; sporadyczne trafienia nie uzasadniają kosztu dodatkowej fuzji przy tych danych.'),'']
    rankings=merged[(merged.family!='all')&(merged.stratum=='all')&(merged.variant=='selected')]
    best=rankings.loc[rankings.groupby('family').miou.idxmax()]
    lines+=table(best,{'Rodzina':'family','Najlepszy selected model':'model','mIoU':'miou','AUC':'auc','AP':'ap'})
    lines+=['','## Materiały i ograniczenia','',
            '[Galeria przypadków](results/gallery/index.html), [manifest wyboru przykładów](results/gallery/cases.json), [komplementarność](results/complementarity_summary.csv), [pełne przedziały powierzchni](results/summary_by_family_area.csv). Kategorie obejmują małe regiony, otwory, poprawy, regresje, wysoką pewność z błędem i niejednoznaczną orientację. Bez GT manipulations nie można rozstrzygnąć, czy model błędnie wykrył edycję, czy poprawnie wykrył edycję niezwiązaną z watermarkiem.',
            '', 'TruFor dostarcza confidence i global detection score. Mesorch i użyty interfejs DetectiveSAM nie udostępniają osobnej skalibrowanej mapy pewności; nie zastępujemy jej arbitralną funkcją prawdopodobieństwa. Panele „confidence_wrong_or_unavailable” w tych modelach nie oznaczają zweryfikowanej wysokiej pewności.',
            '', 'Modele mogły widzieć źródłowe fotografie w swoim pretrainingu; nie mamy identyfikatorów źródeł konkursowych umożliwiających audyt tego nakładania. Nie zmieniono dotychczasowych checkpointów ani wyników. Kod oficjalny i checkpointy przechowywane w ignorowanym results; same źródła eksperymentu są wersjonowalne.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'gallery/cases.json',all_cases)
    html='<!doctype html><meta charset="utf-8"><title>Forensic localization cases</title><style>body{font:16px sans-serif;background:#17191d;color:#eee}img{width:100%}section{margin:30px 0}</style><h1>OOF forensic localization</h1>'
    for c in all_cases:html+=f'<section><p>{c["model"]} / {c["category"]} / {c["image_id"]}</p><img loading="lazy" src="{c["file"]}"></section>'
    (OUT/'gallery/index.html').write_text(html,encoding='utf-8')
    save(OUT/'summary.json',dict(status='fusion_required' if passed else 'complete_no_go',models=models,gates=gates,
                                metrics=merged.to_dict('records'),complementarity=complements,seed=42,n_unique=510,
                                fusion='pending' if passed else 'not_trained_gate_failed'))
    print('Report generated',models,'passed',passed,flush=True)

if __name__=='__main__':main()
