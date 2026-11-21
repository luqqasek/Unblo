"""Measured-only report, comparison tables, seed variability and prediction gallery."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from evaluate import aggregate

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
ROOT=HERE.parents[1]
LABELS={'A':'Extractor','B':'RGB','C':'RGB+SRM','D':'Extractor+RGB+SRM'}


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def table(rows,columns):
    lines=['| '+' | '.join(label for key,label in columns)+' |',
           '| '+' | '.join('---' for _ in columns)+' |']
    for r in rows:
        def fmt(v):return f'{v:.4f}' if isinstance(v,float) else str(v) if v is not None else '—'
        lines.append('| '+' | '.join(fmt(r.get(key)) for key,label in columns)+' |')
    return lines


def heatmap(p):
    # Fixed [0,1] color mapping, never normalize each image independently.
    import matplotlib
    return (matplotlib.colormaps['viridis'](p)[...,:3]*255).astype('uint8')


def visuals(runs,rows):
    gallery=OUT/'gallery';gallery.mkdir(exist_ok=True)
    predictions={}
    for run in runs:
        stem=f'{run["method"]}_{run["variant"]}_fold{run["fold"]}_seed{run["seed"]}'
        z=np.load(OUT/f'{stem}.npz')
        destination=OUT/'heatmaps'/stem;destination.mkdir(parents=True,exist_ok=True)
        for name,p,mask,gt in zip(z['names'],z['probability'],z['mask'],z['target']):
            # Numeric probabilities and binary masks for every seed also live in NPZ.
            Image.fromarray(heatmap(p)).save(destination/f'{name}_heatmap.png')
            Image.fromarray(mask.astype('uint8')*255).save(destination/f'{name}_mask.png')
            if run['seed']==42:predictions[(run['method'],str(name),run['variant'])]=(p,mask,gt)
    arows=[r for r in rows if r['seed']==42 and r['variant']=='A']
    links=[]
    for r in arows:
        method,name=r['method'],r['name']
        rgb=Image.open(ROOT/'data/train/images'/f'{name}.png').convert('RGB').resize((128,128))
        gt=predictions[(method,name,'A')][2]
        panel=Image.new('RGB',(10*132,178),'white');draw=ImageDraw.Draw(panel)
        panel.paste(rgb,(0,28));draw.text((3,8),'RGB',fill='black')
        panel.paste(Image.fromarray(gt.astype('uint8')*255).convert('RGB'),(132,28));draw.text((135,8),'GT',fill='black')
        for j,v in enumerate('ABCD'):
            x=(2+2*j)*132;draw.text((x+3,8),f'{v}: heatmap / mask',fill='black')
            item=predictions.get((method,name,v))
            if item:
                p,mask,_=item
                panel.paste(Image.fromarray(heatmap(p)),(x,28))
                panel.paste(Image.fromarray(mask.astype('uint8')*255).convert('RGB'),(x+132,28))
            else:draw.text((x+3,70),'not promoted',fill='black')
        draw.text((3,160),f'{method} {name} | fold {r["fold"]} | GT {r["area"]:.2%} | seed42 | fixed heatmap scale [0,1]',fill='black')
        filename=f'{method}_{name}.jpg';panel.save(gallery/filename,quality=88)
        links.append(f'<article data-method="{method}"><p>{html.escape(method+" "+name)} · GT {r["area"]:.2%} · fold {r["fold"]}</p><img loading="lazy" src="{filename}"></article>')
    gallery.joinpath('index.html').write_text('<!doctype html><meta charset="utf-8"><title>RGB/SRM OOF</title><style>body{font:16px system-ui;background:#eee}img{width:100%;max-width:1320px}article{margin:20px 0}select{padding:8px}</style><h1>RGB / SRM — seed42 OOF</h1><p>Every evaluated A image; heatmaps use [0,1]. B/C absent outside screening when the gate failed.</p><select onchange="document.querySelectorAll(\'article\').forEach(e=>e.hidden=this.value&&e.dataset.method!==this.value)"><option value="">All methods</option>'+''.join(f'<option>{m}</option>' for m in ('pixelseal','trustmark','stegastamp'))+'</select>'+''.join(links),encoding='utf-8')
    # Representative panels are chosen by GT size, never by favorable new-model gains.
    for method in ('pixelseal','trustmark','stegastamp'):
        subset=sorted([r for r in arows if r['method']==method and r['area']<=.05],key=lambda r:r['area'])
        if not subset:continue
        chosen=[subset[i] for i in sorted(set([0,len(subset)//2,len(subset)-1]))]
        combined=Image.new('RGB',(1320,178*len(chosen)),'white')
        for i,r in enumerate(chosen):combined.paste(Image.open(gallery/f'{method}_{r["name"]}.jpg'),(0,178*i))
        combined.save(OUT/f'{method}_small_examples.png')


def main():
    runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUT.glob('*_fold*_seed*.json'))]
    rows=[dict(method=r['method'],variant=r['variant'],fold=r['fold'],seed=r['seed'],
               epoch=r['epoch'],threshold=r['threshold'],**image) for r in runs for image in r['per_image']]
    write_csv(OUT/'per_image_metrics.csv',rows)
    foldrows=[dict(method=r['method'],variant=r['variant'],fold=r['fold'],seed=r['seed'],
                   epoch=r['epoch'],threshold=r['threshold'],parameters=r['parameters_trainable'],seconds=r['seconds'],
                   **r['metrics']) for r in runs]
    write_csv(OUT/'fold_metrics.csv',foldrows)
    summaries=[];seedrows=[];deltas=[]
    frozen=json.loads((OUT/'extractor_parameters.json').read_text())
    for method,variant in sorted({(r['method'],r['variant']) for r in runs}):
        selected=[r for r in rows if r['method']==method and r['variant']==variant]
        rr=[r for r in runs if r['method']==method and r['variant']==variant]
        full=len(rr)==9 and {(r['fold'],r['seed']) for r in rr}=={(f,s) for f in (1,2,3) for s in (42,7,123)}
        base=dict(method=method,variant=variant,protocol='3-fold x 3 seeds' if full else 'screen fold1 seed42',
                  unique_images=len({r['name'] for r in selected}),runs=len(rr),parameters=rr[0]['parameters_trainable'],
                  seconds=sum(r['seconds'] for r in rr),**aggregate(selected))
        base['frozen_extractor_parameters']=frozen[method]['frozen_extractor'] if variant in ('A','D') else 0
        base['frozen_feature_backbone_parameters']=frozen[method]['feature_backbone'] if variant in ('A','D') else 0
        base['total_parameters']=base['parameters']+base['frozen_extractor_parameters']
        seeds=[]
        for seed in sorted({r['seed'] for r in selected}):
            values=[r for r in selected if r['seed']==seed]
            entry=dict(method=method,variant=variant,seed=seed,**aggregate(values))
            seedrows.append(entry);seeds.append(entry)
        for metric in ('miou','ap','auc','small_ap','small_topk','small_fg_iou','large_clean_iou'):
            base[metric+'_seed_sd']=float(np.std([s[metric] for s in seeds],ddof=1)) if len(seeds)>1 else None
            foldmeans=[np.mean([r['metrics'][metric] for r in rr if r['fold']==f]) for f in sorted({r['fold'] for r in rr})]
            base[metric+'_fold_sd']=float(np.std(foldmeans,ddof=1)) if len(foldmeans)>1 else None
        base['small_unique_images']=len({r['name'] for r in selected if 0<r['area']<=.05})
        summaries.append(base)
        if full and variant!='A':
            controls={(r['name'],r['seed']):r for r in rows if r['method']==method and r['variant']=='A'}
            for scope in ('all','folds2_3','small'):
                scope_rows=[r for r in selected if scope=='all' or (scope=='folds2_3' and r['fold']!=1) or (scope=='small' and 0<r['area']<=.05)]
                for metric in ('miou','ap','auc','topk','fg_iou','clean_iou'):
                    by_image={}
                    for r in scope_rows:by_image.setdefault(r['name'],[]).append(r[metric]-controls[(r['name'],r['seed'])][metric])
                    values=np.array([np.mean(v) for v in by_image.values()])
                    rng=np.random.default_rng(20261121)
                    ci=np.quantile(values[rng.integers(len(values),size=(2000,len(values)))].mean(1),[.025,.975])
                    deltas.append(dict(method=method,variant=variant,scope=scope,metric=metric,delta=float(values.mean()),
                                       ci_low=float(ci[0]),ci_high=float(ci[1]),unique_images=len(values)))
    write_csv(OUT/'comparison.csv',summaries);write_csv(OUT/'seed_metrics.csv',seedrows);write_csv(OUT/'paired_deltas.csv',deltas)
    complete=(OUT/'completion.json').exists()
    save(OUT/'summary.json',dict(status='complete' if complete else 'partial',comparisons=summaries,seeds=seedrows,folds=foldrows,deltas=deltas))
    columns=[('method','Metoda'),('variant','Wariant'),('protocol','Protokół'),('miou','OOF mIoU'),('auc','AUC macro'),('ap','AP macro'),('small_ap','AP ≤5%'),('small_topk','Top-K ≤5%'),('small_fg_iou','FG IoU ≤5%')]
    lines=['# RGB / SRM: wyniki eksperymentu','',
           '**Status: '+('pełny zaplanowany screening i potwierdzenie ukończone.' if complete else 'wyniki częściowe; trening trwa.')+'**','',
           'A = zamrożony ekstraktor + dotychczasowy Head; B = RGB; C = RGB + stałe residuals SRM-style; '
           'D = ekstraktor + RGB/SRM. W StegaStamp A/D używają poprawionego inverse-STN. '
           'Wyniki są zmierzone na GPU, bez danych syntetycznych, testu konkursowego i submission.','',
           '## Porównanie','']+table(summaries,columns)
    conclusions=HERE/'CONCLUSIONS.md'
    if complete and conclusions.exists():
        lines[5:5]=[conclusions.read_text(encoding='utf-8'),'']
    lines+=['','## Screening: identyczny fold1 i seed42','']
    lines+=table([r for r in foldrows if r['fold']==1 and r['seed']==42],
                 [(k,l) for k,l in columns if k!='protocol'])
    lines+=['','Wiersze screeningowe mają tylko fold1/seed42 i nie są porównywalne z pełnym OOF. '
            'Wyniki pełne to średnia po obrazach OOF i trzech seedach, nie ensemble probabilities. '
            'Każda metoda ma 170 unikalnych obrazów; seedy nie zwiększają liczby niezależnych przykładów.','',
            '## Kontrole lokalizacji i dominującej klasy','']
    lines+=table(summaries,[('method','Metoda'),('variant','Wariant'),('fg_iou','FG IoU'),('clean_iou','Clean IoU'),
                           ('large_clean_iou','Clean IoU GT>50%'),('empty_miou','Empty mIoU'),('constant_miou','Decyzja całym obrazem mIoU'),
                           ('random_ap','Losowe AP'),('small_random_ap','Losowe AP ≤5%'),('small_random_topk','Losowe Top-K ≤5%')])
    lines+=['','AUC/AP są średnimi metryk liczonych oddzielnie dla każdego obrazu. Top-K zna prawdziwą '
            'liczbę dodatnich pikseli: jest diagnostyką oracle, nie maską deployable. Remisy Top-K są '
            'rozwiązywane przez oczekiwaną precyzję losowego wyboru wśród remisów. Losowy ranking ma AUC 0.5 '
            'i oczekiwane Top-K równe GT area; AP wykorzystuje dokładną wartość oczekiwaną dla losowej permutacji '
            '(bliską prevalence). Empty to faktyczna pusta maska z oficjalną obsługą pustej unii. '
            'Decyzja całym obrazem: pełna maska gdy średnia probability ≥ wybrany inner próg, inaczej pusta.','',
            '## Zmienność i budżet','']
    lines+=table(summaries,[('method','Metoda'),('variant','Wariant'),('parameters','Parametry trenowane'),
                           ('frozen_extractor_parameters','Parametry zamrożone'),('seconds','Sekundy GPU jobs'),
                           ('miou_seed_sd','mIoU SD seedów'),('miou_fold_sd','mIoU SD foldów'),
                           ('small_ap_seed_sd','Small AP SD seedów'),('small_topk_seed_sd','Small Top-K SD seedów')])
    lines+=['','Budżet optymalizacji jest taki sam: batch8, maks.50 epok, patience8, AdamW lr0.001/decay0.0001, '
            '0.5BCE+0.5Dice; epoka i próg wybierane identycznie wewnątrz train. Czasy obejmują wybór, '
            'refit, ocenę i zapis checkpointu; nie obejmują historycznej ekstrakcji cech. '
            'PS/TM używają dotychczasowego holdoutu80/20, StegaStamp trzech wewnętrznych foldów. '
            'Większy koszt SS nie jest dodatkowym budżetem tylko dla nowej architektury. '
            'GPU RTX3070 i GTX1080Ti, float32 bez TF32. Różne liczby parametrów i operacje GPU oznaczają, '
            'że to porównanie reprezentacji przy tym samym algorytmie treningu, nie ścisły capacity/FLOP match.','',
            'Parametry zamrożone obejmują cały ekstraktor/dekoder (StegaStamp także STN), bez enkodera. '
            'W CSV dodatkowo liczba parametrów backbone potrzebnego do czterech cech, bez końcowego '
            'odczytu wiadomości. Cache cech eliminuje koszt ekstraktora w pętli treningowej. '
            'Modele uczono na dwóch różnych kartach; brak ścisłej deterministyczności między sprzętem '
            'i CPU. Dlatego podstawą porównania jest nowy A na GPU, a nie historyczne wyniki CPU.','',
            '## Screening i potwierdzenie','']
    if (OUT/'screening.json').exists():
        gates=json.loads((OUT/'screening.json').read_text())
        lines+=table([dict(method=g['method'],variant=g['variant'],promoted=g['passed']) for g in gates],
                     [('method','Metoda'),('variant','Wariant'),('promoted','Awans')])
    lines+=['','Kryterium zapisane przed treningiem: poprawa obu AP i Top-K małych masek względem A '
            'oraz macro AP ≥ A−0.01. Nie wybierano modelu na podstawie globalnego mIoU. '
            'D zawsze przechodzi pełne CV. Poniżej delty D−A na foldach2/3, poza screeningiem:','']
    lines+=table([r for r in deltas if r['variant']=='D' and r['scope']=='folds2_3'],
                 [('method','Metoda'),('metric','Metryka'),('delta','Delta D−A'),('ci_low','CI95 low'),('ci_high','CI95 high')])
    lines+=['','Bootstrap: 2000 parowanych losowań obrazów, po uśrednieniu różnic po seedach. '
            'Przedziały warunkowe nie obejmują niepewności treningu ani adaptacyjnego wykorzystania '
            'tego zbioru w poprzednich eksperymentach. SD foldów i seedów opisowe.','',
            '## Artefakty','',
            '[Tabela porównań](results/comparison.csv), [wszystkie obrazy i metryki](results/per_image_metrics.csv), '
            '[foldy](results/fold_metrics.csv), [seedy](results/seed_metrics.csv), '
            '[parowane różnice](results/paired_deltas.csv), [konfiguracja](results/config.json), '
            '[galeria seed42](results/gallery/index.html). Pełne float32 heatmaps i maski dla wszystkich '
            'seedów w plikach NPZ; ich PNG w results/heatmaps/.','',
            '[Weryfikacja](results/verification.json): wszystkie 60 checkpointów odtworzyło nowe predykcje '
            'dokładnie na GPU; historyczne checkpointy CPU sprawdzone na GPU z maksymalnym błędem '
            'około 1.1e−6. Oficjalne mIoU zweryfikowane na wszystkich obrazach, a splity i wybór '
            'epoki/progu sprawdzone z zapisanymi historiami. Hashe źródeł i wcześniejszych najlepszych '
            'artefaktów pozostały niezmienione.','',
            '## Ograniczenia','',
            'Mały zbiór (170 obrazów/metodę), istniejące wielokrotnie analizowane foldy. Screening '
            'nie jest niezależnym testem. B/C bez awansu nie są ocenione na pozostałych foldach '
            'i nie uzasadniają kategorycznego wniosku o wszystkich RGB/SRM architekturach. '
            'Użyto trzech filtrów, konkretnej lekkiej CNN i wejścia256; nie pełnego banku SRM '
            'ani strojenia rozdzielczości. D ma większą liczbę parametrów niż A; bez kontroli capacity '
            'nie przypisujemy każdej różnicy samym residuals. Zachowano najlepsze wcześniejsze rozwiązanie.','']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Rebuilding prose/metrics need not rewrite thousands of unchanged prediction PNGs.
    if not (OUT/'gallery/index.html').exists():visuals(runs,rows)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
