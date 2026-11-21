"""Measured, paired scaling effects; separate data diversity from update budget."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scaling_common import *

analysis=module('scaling_metric_definitions',OLD/'report.py')
METRICS=['miou','fg_iou','clean_iou','auc','ap','topk','delta_empty','pred_area','probability_separation','brier']
LABELS={'real_only':'Tylko real','n2k_s2500':'2k / 2500 kroków','n4k_s2500':'4k / 2500 kroków',
        'n2k_s5000':'2k / 5000 kroków','n4k_s5000':'4k / 5000 kroków'}
CONTRASTS=[('synthetics_2k','real_only','n2k_s2500'),('more_data_2500','n2k_s2500','n4k_s2500'),
           ('more_data_5000','n2k_s5000','n4k_s5000'),('more_updates_2k','n2k_s2500','n2k_s5000'),
           ('more_updates_4k','n4k_s2500','n4k_s5000'),('practical_double','n2k_s2500','n4k_s5000')]

def measured_run(path,meta):
    cache=OUT/'metric_chunks'/path.parent.name/path.with_suffix('.json').name
    identity=dict(prediction_sha256=sha(path),metric_code_sha256=sha(OLD/'report.py'),run_metadata_sha256=sha(path.with_suffix('.json')))
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved['identity']==identity:return saved['rows']
    r=json.loads(path.with_suffix('.json').read_text());z=np.load(path);rows=[]
    for name,p,g in zip(z['names'],z['probability'],z['target']):
        rows.append(dict(method=r['method'],variant=r['variant'],seed=r['seed'],fold=r['fold'],image_id=str(name),
                         full_watermark=meta[str(name)]['full_watermark'],**analysis.measure(p,g,float(z['threshold']))))
    save(cache,dict(identity=identity,rows=rows))
    return rows

def collect():
    sources=sorted((OUT/'runs').glob('*/*_fold*_seed*.npz'))
    assert len(sources)==135, f'Expected 135 complete outer fits, got {len(sources)}'
    hashes={str(p.relative_to(OUT)):sha(p) for p in sources}
    manifest=OUT/'metrics_cache.json';csv=OUT/'per_image_metrics.csv'
    meta={r['image_name']:r for r in json.loads((ROOT/'data/train/metadata.json').read_text())}
    if csv.exists() and manifest.exists() and json.loads(manifest.read_text())==hashes:
        frame=pd.read_csv(csv)
    else:
        rows=[]
        for j,path in enumerate(sources):
            rows.extend(measured_run(path,meta))
            if (j+1)%15==0:print('REAL METRICS',j+1,'/135',flush=True)
        frame=pd.DataFrame(rows);frame.to_csv(csv,index=False);save(manifest,hashes)
    assert len(frame)==7650
    runs=[json.loads(p.with_suffix('.json').read_text()) for p in sources]
    return frame,runs

def transfer():
    sources=sorted((OUT/'runs').glob('*/*_synthetic_only.npz'));assert len(sources)==36
    hashes={str(p.relative_to(OUT)):sha(p) for p in sources}
    path=OUT/'synthetic_only_real_metrics.csv';cache=OUT/'transfer_cache.json'
    if path.exists() and cache.exists() and json.loads(cache.read_text())==hashes:return pd.read_csv(path)
    rows=[]
    for path in sources:
        method=path.parent.name;parts=path.stem.split('_');variant='_'.join(parts[:2]);seed=int(parts[2][4:])
        z=np.load(path)
        for name,p,g in zip(z['names'],z['probability'],z['target']):
            rows.append(dict(method=method,variant=variant,seed=seed,image_id=str(name),**analysis.measure(p,g,.5)))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'synthetic_only_real_metrics.csv',index=False);save(cache,hashes)
    return frame

def summarize(frame):
    rows=[]
    for (method,variant),data in frame.groupby(['method','variant']):
        small=data[data.gt_area<=.05];large=data[data.gt_area>.5]
        row=dict(method=method,variant=variant,n=data.image_id.nunique(),seeds=3,protocol='3-fold CV × 3 seeds; after real finetuning',
                 **{m:float(data[m].mean()) for m in METRICS},
                 seed_sd=float(data.groupby('seed').miou.mean().std()),
                 small_n=small.image_id.nunique(),large_n=large.image_id.nunique(),
                 small_miou=float(small.miou.mean()),small_fg_iou=float(small.fg_iou.mean()),small_ap=float(small.ap.mean()),
                 small_auc=float(small.auc.mean()),small_delta_empty=float(small.delta_empty.mean()),
                 large_clean_iou=float(large.clean_iou.mean()))
        rows.append(row)
    return rows

def effects(frame):
    means=frame.groupby(['method','variant','image_id'])[METRICS].mean()
    comparisons=[];paired=[]
    for method in METHODS:
        for label,base,variant in CONTRASTS:
            a=means.loc[(method,base)];b=means.loc[(method,variant)].reindex(a.index)
            metrics={}
            for m in METRICS:
                d=b[m]-a[m]
                metrics[m]=dict(base=float(a[m].mean()),variant=float(b[m].mean()),delta=float(d.mean()),ci95=analysis.ci(d))
            sub=frame[frame.method==method]
            per_seed=[];pairs=[]
            for seed in (42,7,123):
                aa=sub[(sub.variant==base)&(sub.seed==seed)];bb=sub[(sub.variant==variant)&(sub.seed==seed)]
                per_seed.append(dict(seed=seed,base=float(aa.miou.mean()),variant=float(bb.miou.mean()),delta=float(bb.miou.mean()-aa.miou.mean())))
                for fold in (1,2,3):pairs.append(float(bb[bb.fold==fold].miou.mean()-aa[aa.fold==fold].miou.mean()))
            comparisons.append(dict(method=method,contrast=label,base=base,variant=variant,metrics=metrics,
                                    positive_fold_seed_pairs=sum(v>0 for v in pairs),seeds=per_seed,n=170))
            for name in a.index:
                paired.append(dict(method=method,contrast=label,image_id=name,**{'delta_'+m:float(b.loc[name,m]-a.loc[name,m]) for m in METRICS}))
    pd.DataFrame(paired).to_csv(OUT/'paired_deltas.csv',index=False)
    return comparisons

def figures(frame,runs):
    contrasts=json.loads((OUT/'summary.json').read_text())['effects']
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    chosen=[('more_data_2500','4k − 2k / 2500 steps'),('more_data_5000','4k − 2k / 5000 steps'),
            ('more_updates_2k','5000 − 2500 steps / 2k'),('practical_double','4k/5000 − 2k/2500')]
    for ax,method in zip(axes,METHODS):
        for y,(key,label) in enumerate(chosen):
            r=next(r for r in contrasts if r['method']==method and r['contrast']==key)['metrics']['miou']
            ax.plot(r['ci95'],[y,y],color='#247ba0',linewidth=2);ax.plot(r['delta'],y,'o',color='#247ba0')
        ax.axvline(0,color='gray',linestyle='--');ax.set_title(method);ax.set_yticks(range(len(chosen)),[label for _,label in chosen])
        ax.set_xlabel('Paired mIoU delta with 95% CI');ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();fig.tight_layout();fig.savefig(OUT/'paired_effects.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),squeeze=False)
    for col,method in enumerate(METHODS):
        data=frame[frame.method==method]
        for row,metric in enumerate(('miou','fg_iou')):
            ax=axes[row,col]
            baseline=data[data.variant=='real_only'].groupby('seed')[metric].mean()
            ax.axhline(baseline.mean(),color='gray',linestyle='--',label='Tylko real')
            for steps,color in ((2500,'#247ba0'),(5000,'#e07a36')):
                groups=[data[data.variant==f'n{n}k_s{steps}'].groupby('seed')[metric].mean() for n in (2,4)]
                ax.errorbar([2000,4000],[g.mean() for g in groups],yerr=[g.std() for g in groups],marker='o',capsize=4,color=color,label=f'{steps} kroków')
            ax.set(title=method,xlabel='Liczba różnych syntetyków',ylabel=metric,xticks=[2000,4000]);ax.legend()
    fig.suptitle('Real OOF po fine-tuningu; error bars = SD trzech seedów (nie CI)',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'scaling_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,method in zip(axes,METHODS):
        for n,color in ((2000,'#247ba0'),(4000,'#e07a36')):
            histories=[]
            for seed in (42,7,123):
                variant='n2k_s5000' if n==2000 else 'n4k_s5000'
                h=json.loads((OUT/'runs'/method/f'{variant}_seed{seed}_pretrain.json').read_text())['history']
                histories.append([r['synthetic_validation_miou'] for r in h])
            values=np.asarray(histories);x=np.arange(1,values.shape[1]+1)*n/8
            ax.plot(x,values.mean(0),color=color,label=f'{n} zdjęć');ax.fill_between(x,values.mean(0)-values.std(0),values.mean(0)+values.std(0),color=color,alpha=.15)
        ax.set(title=method,xlabel='Aktualizacje optymalizatora',ylabel='Syntetyczna walidacja mIoU');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'pretrain_learning.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,method in zip(axes,METHODS):
        data=frame[(frame.method==method)&(frame.gt_area<=.05)]
        for metric,marker in [('ap','o'),('fg_iou','s')]:
            values=[data[data.variant==v][metric].mean() for v in VARIANTS]
            ax.plot(range(5),values,marker=marker,label=metric)
        ax.set(title=f'{method} / GT ≤5%',xticks=range(5),xticklabels=['real','2k/2.5k','4k/2.5k','2k/5k','4k/5k']);ax.tick_params(axis='x',rotation=25);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'small_regions.png',dpi=150);plt.close(fig)

def gallery(frame):
    out=OUT/'gallery';out.mkdir(exist_ok=True);sections=[]
    for method in METHODS:
        avg=frame[frame.method==method].groupby(['variant','image_id']).miou.mean().unstack('variant')
        delta=(avg.n4k_s5000-avg.n2k_s2500).sort_values()
        small=frame[(frame.method==method)&(frame.variant=='n4k_s5000')&(frame.seed==42)&(frame.gt_area<=.05)].sort_values('fg_iou')
        chosen=list(dict.fromkeys(list(delta.head(3).index)+list(delta.tail(3).index)+list(small.head(1).image_id)+list(small.tail(1).image_id)))
        maps={}
        for variant in ('real_only','n2k_s2500','n4k_s5000'):
            for fold in (1,2,3):
                z=np.load(OUT/'runs'/method/f'{variant}_fold{fold}_seed42.npz')
                for name,p,g in zip(z['names'],z['probability'],z['target']):maps[variant,str(name)]=(p,g.astype(bool),float(z['threshold']))
        sections.append(f'<h2 id="{method}">{method}</h2>')
        for name in chosen:
            rgb=np.asarray(Image.open(ROOT/'data/train/images'/f'{name}.png').convert('RGB'))
            p0,g,t0=maps['real_only',name];p2,_,t2=maps['n2k_s2500',name];p4,_,t4=maps['n4k_s5000',name]
            err=np.zeros((*g.shape,3));err[(p4>=t4)&~g]=[1,0,0];err[(p4<t4)&g]=[0,0,1]
            fig,axs=plt.subplots(1,8,figsize=(22,3))
            for ax,im,title in zip(axs,[rgb,g,p0,p2,p4,p2>=t2,p4>=t4,err],['RGB','GT','P real','P 2k/2500','P 4k/5000',f'2k t={t2}',f'4k t={t4}','FP red / FN blue']):
                ax.imshow(im,cmap='viridis' if title.startswith('P ') else 'gray',vmin=0 if im.ndim==2 else None,vmax=1 if im.ndim==2 else None);ax.set_title(title);ax.axis('off')
            m2=analysis.measure(p2,g,t2);m4=analysis.measure(p4,g,t4)
            fig.suptitle(f'{method} {name} seed42 | mIoU {m2["miou"]:.3f}→{m4["miou"]:.3f} | FG {m2["fg_iou"]:.3f}→{m4["fg_iou"]:.3f} | AP {m2["ap"]:.3f}→{m4["ap"]:.3f} | GT {g.mean():.3f}')
            fig.tight_layout();filename=f'{method}_{name}.jpg';fig.savefig(out/filename,dpi=100);plt.close(fig)
            sections.append(f'<p>{name}</p><img loading="lazy" src="{filename}" style="width:100%">')
    (out/'index.html').write_text('<!doctype html><html lang="pl"><meta charset="utf-8"><title>Syntetyki: skalowanie</title><body style="font-family:system-ui;margin:24px"><h1>Skalowanie syntetyków — przykłady real OOF</h1><p>Trzy największe pogorszenia/poprawy 4k5000 vs 2k2500 po średniej seedów oraz małe maski; panele pokazują seed42. Diagnostyczny dobór, nie reprezentatywna próba.</p><nav><a href="#pixelseal">PixelSeal</a> · <a href="#trustmark">TrustMark</a> · <a href="#stegastamp">StegaStamp</a></nav>'+''.join(sections)+'</body></html>',encoding='utf-8')

def main():
    assert all((OUT/'runs'/m/'complete.json').exists() for m in METHODS)
    frame,runs=collect();comparisons=summarize(frame);contrasts=effects(frame);zero=transfer()
    zero_summary=zero.groupby(['method','variant'])[METRICS].mean().reset_index()
    zero_summary.to_csv(OUT/'synthetic_only_summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'summary_metrics.csv',index=False)
    pd.DataFrame([{k:r[k] for k in ('method','variant','seed','fold','epoch','threshold','miou','fg_iou','clean_iou','seconds')} for r in runs]).to_csv(OUT/'fold_metrics.csv',index=False)
    frame['size_bin']=pd.cut(frame.gt_area,[-1e-9,0,.01,.05,.2,.5,1.000001],labels=['empty','0–1%','1–5%','5–20%','20–50%','50–100%'])
    strata=[]
    for (method,variant,size),f in frame.groupby(['method','variant','size_bin'],observed=True):
        strata.append(dict(method=method,variant=variant,size_bin=str(size),n=f.image_id.nunique(),unreliable=f.image_id.nunique()<10,**{m:float(f[m].mean()) for m in METRICS}))
    pd.DataFrame(strata).to_csv(OUT/'size_stratified_metrics.csv',index=False)
    frame.groupby(['method','variant','full_watermark'])[METRICS].mean().to_csv(OUT/'embedding_mode_metrics.csv')
    seed_rows=frame.groupby(['method','variant','seed'])[METRICS].mean().reset_index();seed_rows.to_csv(OUT/'seed_metrics.csv',index=False)
    half_scores={}
    for method in METHODS:
        for variant in VARIANTS:
            selected=[r for r in runs if r['method']==method and r['variant']==variant]
            half_scores[method,variant]=sum(r['fixed_half_miou']*len(r['names']) for r in selected)/sum(len(r['names']) for r in selected)
    summary=dict(status='complete',real_unique_images=510,outer_fits=135,methods=list(METHODS),comparisons=comparisons,effects=contrasts,
                 seed_metrics=seed_rows.to_dict('records'),synthetic_only=zero_summary.to_dict('records'),
                 finetuned_fixed_half=[dict(method=m,variant=v,miou=value) for (m,v),value in half_scores.items()],
                 folds=[{k:r[k] for k in ('method','variant','seed','fold','epoch','threshold','miou','fg_iou','clean_iou','seconds')} for r in runs],
                 protocol='Primary: post-finetune real OOF. Nested 2k/4k covers × equal 2500/5000 updates; three seeds and unchanged outer/inner splits.',
                 uncertainty='5000 paired image bootstrap resamples after seed averaging; conditional on fitted models; development CV, multiple comparisons uncorrected.')
    costs=[]
    for method in METHODS:
        # Milestone snapshots contain cumulative time: count only each trajectory's final snapshot.
        pretrain=sum(json.loads((OUT/'runs'/method/f'{v}_seed{s}_pretrain.json').read_text())['seconds']
                     for v in ('n2k_s5000','n4k_s5000') for s in (42,7,123))
        finetune=sum(r['seconds'] for r in runs if r['method']==method)
        generation=json.loads((OUT/'data'/method/'complete.json').read_text())['seconds']
        costs.append(dict(method=method,pretrain_compute_seconds=pretrain,inner_and_outer_fit_seconds=finetune,generation_seconds=generation,
                          note='Pretrain timers exclude checkpoint I/O; SS generation reuses prior 2200 examples. Parallel execution means summed times are not elapsed wall time.'))
    save(OUT/'compute_cost.json',costs)
    save(OUT/'summary.json',summary);figures(frame,runs);gallery(frame)
    rows={(r['method'],r['variant']):r for r in comparisons}
    get=lambda m,c:next(r for r in contrasts if r['method']==m and r['contrast']==c)
    lines=['# Syntetyki dla wszystkich rodzin: 2k vs 4k', '',
           '**Wykonano cały zaplanowany eksperyment: 3 metody × 5 wariantów × 3 seedy × 3 foldy = 135 końcowych fitów OOF.** Wszystkie poniższe główne wyniki dotyczą modeli po fine-tuningu na realnych danych. 510 unikalnych obrazów, po 170 na metodę; powtórzenia seedów nie zwiększają liczby niezależnych zdjęć.', '',
           '## Odpowiedź na pytania', '', '| Metoda | Tylko real | 2k / 2500 kroków | 4k / 2500 kroków | 2k / 5000 kroków | 4k / 5000 kroków |', '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:lines.append('| '+method+' | '+' | '.join(f'{rows[method,v]["miou"]:.5f}' for v in VARIANTS)+' |')
    lines+=['', 'Główna miara: oficjalna średnia per-image dwuklasowego mIoU. Różne rodziny mają odrębne modele; nie porównujemy ich jako jednego modelu.', '']
    for method in METHODS:
        base=get(method,'synthetics_2k')['metrics']['miou'];fixed=get(method,'more_data_2500')['metrics']['miou'];long=get(method,'more_data_5000')['metrics']['miou'];practical=get(method,'practical_double')['metrics']['miou']
        lines += [f'- **{method}:** 2k względem real-only {base["delta"]:+.5f}. Podwojenie liczby danych przy 2500 krokach: {fixed["delta"]:+.5f}, przy 5000: {long["delta"]:+.5f}. Praktyczne 2k×10epok → 4k×10epok: {practical["delta"]:+.5f}.']
    lines += ['', 'Sama dodatnia różnica punktowa nie rozstrzyga przewagi. Poniżej parowane przedziały i wyniki seedów, oraz oddzielenie większej puli od dłuższego treningu.', '',
              '## Co dokładnie porównano', '',
              'Pula 4000 zawiera wcześniejszą pulę2000 i 2000 nowych zdjęć. Stałe200 coverów syntetycznej walidacji są wyłączone z obu treningów. Wszystkie źródła COCO z jednej ustalonej permutacji; po jednym syntetyku na źródło i metodę. SS zachowuje dokładne wcześniejsze przykłady oraz inverse-STN. PixelSeal: 256 bit?w, TrustMark Q: 100 bit?w bez ECC — rzeczywiste encodery ze starter kitu.', '',
              'Pretrain: 2k×10epok i4k×5epok mają po 2500 aktualizacji; 2k×20 i4k×10 po 5000. Przy tych samych krokach i batch 8 model widzi tę samą liczbę przykładów, ale różną liczbę unikalnych zdjęć. Cztery checkpointy są z góry określone; syntetyczna walidacja ich nie wybiera.', '',
              'Fine-tuning: niezmieniona głowica i zamrożone ekstraktory; tylko realny outer-train. Istniejący inner80/20 PS/TM lub trzy inner-foldy SS wybierają epokę i próg, następnie świeży refit od właściwego synthetic checkpointu. AdamW .001/.0001, batch 8, 0.5 BCE +0.5 Dice, maks. 50 epok / patience 8, progi .3–.7. Kontrole real-only uczone tym samym kodem i urządzeniem co warianty danej metody.', '',
              'Dla TrustMark bbox przekazywany do encodera ma proporcje <=2:1. Cienka docelowa maska może znajdować się wewnątrz szerszego poprawnie oznakowanego cropa; poza maską przywracamy identyczne czyste piksele. Nie zmieniono architektury ani loss razem z liczebnością danych.', '',
              '[Plan przed wynikami](README.md) · [Instrukcja odtworzenia](REPRODUCE.md). Badanie nie uruchamia predykcji testowych ani zgłoszeń; istniejący najlepszy wynik #1496 pozostaje zachowany.', '',
              '### Kontrola generatora', '',
              'Liczby 2k i 4k oznaczają przykłady treningowe na każdą metodę. Metody korzystają z tych samych zdjęć źródłowych: 4000 treningowych i 200 walidacyjnych coverów, a po zastosowaniu trzech encoderów powstało 12 600 par obraz–maska. Nie traktujemy ich jako 12 600 niezależnych zdjęć źródłowych.', '',
              '![Przykładowe syntetyki i rzeczywisty residual](results/generation_examples.png)', '',
              'Ten sam cover i maska, trzy rzeczywiste encodery. Kolumna residualu pokazuje średnią bezwzględną zmianę RGB, z osobną skalą każdego panelu; nie służy porównaniu siły znakowania.', '',
              '![Rozkład powierzchni masek](results/generation_areas.png)', '',
              'Rozkład powierzchni porównano dla obu pul i realnych obrazów. Synteza zawiera po 5% całkowicie czystych i całkowicie oznakowanych przykładów; realne 170 obrazów każdej metody nie zawiera takich skrajnych masek. To zamierzona kontrola treningowa, ale również różnica domen. Pula 4k ma podobny rozkład powierzchni do 2k. [Liczby](results/generation_area_summary.json) · [Kontrola wszystkich masek i cech](results/data_verification.json).', '',
              '## Efekt danych i efekt budżetu', '',
              '| Metoda | Kontrast | Delta mIoU | CI95 | Dodatnie pary fold–seed |', '|---|---|---:|---|---:|']
    names={'synthetics_2k':'2k2500 − real','more_data_2500':'4k2500 − 2k2500: dane','more_data_5000':'4k5000 − 2k5000: dane',
           'more_updates_2k':'2k5000 − 2k2500: kroki','more_updates_4k':'4k5000 − 4k2500: kroki','practical_double':'4k5000 − 2k2500: oba'}
    for r in contrasts:
        m=r['metrics']['miou'];lines.append(f'| {r["method"]} | {names[r["contrast"]]} | {m["delta"]:+.5f} | [{m["ci95"][0]:+.5f}, {m["ci95"][1]:+.5f}] | {r["positive_fold_seed_pairs"]}/9 |')
    lines += ['', 'CI:5000 bootstrapów170 obrazów danej metody po uśrednieniu seedów, parowane obrazy. Nie obejmują ponownego treningu ani zmienności innych pul syntetycznych. Zbiór rozwojowy był wcześniej wielokrotnie analizowany; wiele kontrastów bez korekty wielokrotnych porównań. Dodatnie foldy nie są niezależnymi testami.', '',
              '![Parowane efekty danych i kroków](results/paired_effects.png)', '',
              '![Skalowanie po fine-tuningu](results/scaling_curves.png)', '',
              '## Czy poprawia się lokalizacja?', '',
              '| Metoda | Wariant | FG IoU | Clean IoU | AUC macro | AP macro | Małe FG IoU | Małe AP | Małe delta vs empty |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for v in VARIANTS:
            r=rows[method,v];lines.append(f'| {method} | {LABELS[v]} | {r["fg_iou"]:.4f} | {r["clean_iou"]:.4f} | {r["auc"]:.4f} | {r["ap"]:.4f} | {r["small_fg_iou"]:.4f} | {r["small_ap"]:.4f} | {r["small_delta_empty"]:+.4f} |')
    small_counts=', '.join(f'{m}: {rows[m,"real_only"]["small_n"]}' for m in METHODS)
    lines += ['', f'Małe maski: GT ≤ 5%; liczba unikalnych obrazów — {small_counts}. AUC/AP są średnimi per obraz, nie pooled. AP to Average Precision. Top-K w CSV używa znanej powierzchni GT i jest tylko diagnostyką. Jeśli mIoU rośnie głównie przez clean IoU, nie oznacza to rozwiązania lokalizacji watermarku.', '',
              '![Małe regiony](results/small_regions.png)', '',
              '[Pełne przedziały powierzchni](results/size_stratified_metrics.csv) · [Tryb znakowania](results/embedding_mode_metrics.csv) · [Wszystkie obrazy](results/per_image_metrics.csv) · [Galeria błędów/popraw](results/gallery/index.html). Biny z n<10 oznaczono jako mało wiarygodne.', '',
              '## Seedy i fine-tuning', '', '| Metoda | Wariant | Seed42 | Seed7 | Seed123 | SD seedów |', '|---|---|---:|---:|---:|---:|']
    for method in METHODS:
        for v in VARIANTS:
            f=seed_rows[(seed_rows.method==method)&(seed_rows.variant==v)].set_index('seed')
            lines.append(f'| {method} | {LABELS[v]} | {f.loc[42,"miou"]:.5f} | {f.loc[7,"miou"]:.5f} | {f.loc[123,"miou"]:.5f} | {rows[method,v]["seed_sd"]:.5f} |')
    lines += ['', 'Pomocnicza kontrola synthetic-only używa stałego progu 0.5 na 170 realnych obrazach, bez uczenia na real. Fine-tuning ma próg dobrany w inner CV; różnica obejmuje więc adaptację i kalibrację. Główne decyzje w tym raporcie odnoszą się do modeli po fine-tuningu, zgodnie z wyborem użytkownika.', '',
              '| Metoda | Wariant | Synthetic-only, próg 0.5 | Po fine-tuningu, próg 0.5 | Po fine-tuningu, próg inner CV |', '|---|---|---:|---:|---:|']
    for method in METHODS:
        for v in VARIANTS[1:]:
            z=zero_summary[(zero_summary.method==method)&(zero_summary.variant==v)].iloc[0]
            lines.append(f'| {method} | {LABELS[v]} | {z.miou:.5f} | {half_scores[method,v]:.5f} | {rows[method,v]["miou"]:.5f} |')
    lines += ['', '![Uczenie syntetyczne](results/pretrain_learning.png)', '',
              '## Wiarygodność i ograniczenia', '',
              'Kontrole obejmują zgodność cech z bazą, dokładny dotychczasowy zbiórSS, poprawny bbox TrustMark, rozłączne źródła train/holdout, SHA/dHash względem real 510, niezmienność istniejących modeli i zgłoszenia, official mask downsampling, OOF name mapping, wewnętrzny wybór parametrów i odtworzenie predykcji z checkpointów. Różnice CPU/GPU cech są zapisane w preflightach; nie nazywamy ich idealną zgodnością bitową.', '',
              'Jeden zewnętrzny zbiór i jedna deterministyczna pula generatora. Dwukrotne zwiększenie do 4k nie odpowiada na pytanie o 10k/100k i nie dowodzi monotonicznej poprawy. Syntetyki nie odtwarzają wszystkich generatywnych edycji konkursowych; maski pochodzą z mieszania ze znanym czystym zdjęciem. Brak identyfikatorów źródeł real ogranicza pewność kontroli near-duplicates.', '',
              '## Wniosek dla dalszych eksperymentów', '']
    for method in METHODS:
        a=get(method,'more_data_2500')['metrics']['miou'];b=get(method,'more_data_5000')['metrics']['miou'];p=get(method,'practical_double')['metrics']['miou']
        if a['ci95'][0]>0 and b['ci95'][0]>0:message='Większa pula poprawia wynik przy obu stałych budżetach; jest uzasadnienie do następnego, większego poziomu danych.'
        elif a['delta']>0 or b['delta']>0:message='Efekt większej puli jest zależny od budżetu lub niepewny; przed dalszym skalowaniem sprawdzić stabilność i foreground metrics.'
        else:message='Brak zmierzonej korzyści z samego podwojenia puli przy tych budżetach; dalsze zwiększanie liczby danych nie jest jeszcze uzasadnione.'
        best=max((rows[method,v] for v in VARIANTS),key=lambda r:r['miou'])
        lines.append(f'- **{method}:** {message} Najwyższa średnia w tej siatce: {LABELS[best["variant"]]} ({best["miou"]:.5f}); jest to wybór po analizie rozwojowej, nie niezależnie potwierdzony zwycięzca.')
    lines += ['', 'Nie przenosić automatycznie jednej receptury na wszystkie rodziny. Przed wdrożeniem sprawdzić, czy zwycięski wariant poprawia ranking i regiony, czy tylko przewiduje więcej czystego tła. Nie zmieniać modelu ani generatora jednocześnie z kolejnym poziomem liczebności.', '',
              '[Wyniki JSON](results/summary.json) · [Metryki CSV](results/summary_metrics.csv) · [Foldy](results/fold_metrics.csv) · [Koszt obliczeń](results/compute_cost.json) · [Weryfikacja](results/verification.json).', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(pd.DataFrame(comparisons)[['method','variant','miou','fg_iou','ap','small_fg_iou']].to_string(index=False),flush=True)

if __name__=='__main__':main()
