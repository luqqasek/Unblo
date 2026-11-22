"""Summarize only completed runs; compare matched images and preserve protocol boundaries."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .prepare import ROOT,HERE,OUT,FAMILIES,save

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def avg(rows,k='miou'):return float(np.mean([r[k] for r in rows]))
def report():
    references=read(HERE/'reference_best.json')
    usage={};source_sets={};real_names=set()
    for family in FAMILIES:
        cache=OUT/'data'/family;records=read(cache/'records.json')
        train=[r for r in records if r['partition']=='train'];dev=[r for r in records if r['partition']=='validation'];real=[r for r in records if r['partition']=='real']
        assert (len(train),len(dev),len(real))==(4000,200,170)
        source_sets[family]={r['source_id'] for r in train}
        assert not source_sets[family]&{r['source_id'] for r in dev}
        assert not real_names&{r['name'] for r in real};real_names.update(r['name'] for r in real)
        usage[family]={'preparation':read(cache/'complete.json'),'train_unique_sources':len(source_sets[family]),'development_unique_sources':len({r['source_id'] for r in dev}),'real_images':len(real)}
        if (cache/'native_bits_complete.json').exists():usage[family]['native_bit_cache']=read(cache/'native_bits_complete.json')
    save(OUT/'data_usage.json',{'families':usage,'real_unique_images':len(real_names),'synthetic_train_unique_source_ids_across_families':len(set.union(*source_sets.values()))})
    result={};measurements=[];paired=[];hardware=[]
    lines=['# Wszystkie informacje — test na RTX 3070','',
      'Porównanie dwóch wariantów kompaktowego modelu na wszystkich 510 oznaczonych obrazach: trzy foldy dla każdej z trzech metod. To walidacja rozwojowa, a nie nowy wynik publiczny. Najlepsze zaakceptowane zgłoszenie #2675 ma publiczne mIoU 0,6440; nowego zgłoszenia nie wykonano.','',
      '## Wyniki','', '| Metoda | Kontrola: mIoU | Wszystkie informacje: mIoU | Zmiana | 95% bootstrap zmiany |','|---|---:|---:|---:|---|']
    for family in FAMILIES:
        folder=OUT/'runs'/family
        if not (folder/'completed.json').exists():continue
        r=read(folder/'completed.json');result[family]=r
        c=read(folder/'control/completed.json')['per_image'];f=read(folder/'all_information/completed.json')['per_image'];cb={x['image_name']:x for x in c};fb={x['image_name']:x for x in f}
        assert set(cb)==set(fb) and len(fb)==170
        ci=r['conditional_bootstrap95']
        lines.append(f"| {family} | {avg(c):.6f} | {avg(f):.6f} | {r['delta']:+.6f} | [{ci[0]:+.6f}; {ci[1]:+.6f}] |")
        for arm,rows in [('control',c),('all_information',f)]:
            folds=[read(folder/arm/f'fold{i}/completed.json') for i in (1,2,3)]
            measurements.append({'family':family,'arm':arm,'miou':avg(rows),'n':170,'folds':[x['summary']['all']['miou'] for x in folds]})
            hardware.append({'family':family,'arm':arm,'peak_gpu_GB':max(x['peak_gpu_bytes'] for x in folds)/1e9,'fold_train_eval_seconds':sum(x['seconds'] for x in folds)})
        for n in sorted(fb):paired.append({'family':family,'image_name':n,'control_miou':cb[n]['miou'],'all_information_miou':fb[n]['miou'],'delta':fb[n]['miou']-cb[n]['miou'],'true_area':fb[n]['true_area'],'fg_iou':fb[n]['fg_iou'],'clean_iou':fb[n]['clean_iou'],'fg_miss':fb[n]['fg_miss'],'clean_miss':fb[n]['clean_miss']})
        ref={x['image_name']:x for x in references['per_image'] if x['family']==family};assert set(ref)<=set(fb)
        r['same_image_best_reference']={'n':len(ref),'reference':avg(list(ref.values())),'all_information':avg([fb[n] for n in ref]),'delta':float(np.mean([fb[n]['miou']-ref[n]['miou'] for n in ref])),'protocol':references['protocols'][family]}
        wrong=[]
        for fold in (1,2,3):wrong+=read(folder/'all_information'/f'fold{fold}/wrong_message.json')['per_image']
        r['wrong_message_diagnostic']={'correct':avg(f),'complemented_message':avg(wrong),'note':'Sensitivity diagnostic against original labels; complemented queries have no independent GT.'}
        # Six image-level examples, selected by largest positive/negative paired deltas.
        order=sorted(cb,key=lambda n:fb[n]['miou']-cb[n]['miou']);chosen=order[:3]+order[-3:]
        probability={}
        for arm in ('control','all_information'):
            for fold in (1,2,3):
                with np.load(folder/arm/f'fold{fold}/validation.npz') as p:
                    for n,a,y in zip(p['names'].astype(str),p['probabilities'],p['targets']):
                        if n in chosen:probability[(arm,n)]=(a,y)
        canvas=Image.new('RGB',(4*256,6*286),'white');draw=ImageDraw.Draw(canvas)
        for j,n in enumerate(chosen):
            a,y=probability['control',n];b,_=probability['all_information',n]
            images=[Image.open(ROOT/'data/train/images'/f'{n}.png').convert('RGB'),Image.fromarray(y.astype(np.uint8)*255),Image.fromarray((a>=.5).astype(np.uint8)*255),Image.fromarray((b>=.5).astype(np.uint8)*255)]
            labels=[n,'GT',f"control {cb[n]['miou']:.4f}",f"all info {fb[n]['miou']:.4f}"]
            for k,(im,title) in enumerate(zip(images,labels)):
                canvas.paste(im.resize((256,256),Image.Resampling.NEAREST).convert('RGB'),(k*256,j*286+25));draw.text((k*256+5,j*286+5),title,fill='black')
        canvas.save(OUT/f'{family}_example_comparison.png')
    assert len(result)==3,'Do not mark an incomplete comparison complete'
    overall={'control_miou':float(np.mean([r['control_miou'] for r in paired])),'all_information_miou':float(np.mean([r['all_information_miou'] for r in paired])),'delta':float(np.mean([r['delta'] for r in paired])),'n':len(paired)}
    lines.append(f"| Wszystkie 510 obrazów | {overall['control_miou']:.6f} | {overall['all_information_miou']:.6f} | {overall['delta']:+.6f} | — |")
    lines+=['','Przedziały bootstrap opisują zmienność obrazów przy tych konkretnych wytrenowanych modelach. Jeden seed; nie obejmują zmienności całego treningu. Foldy były wykorzystywane w poprzednich eksperymentach.','', '## Porównanie z dotychczasowym najlepszym systemem','', '| Metoda | Te same obrazy | Zapisana baza | Nowy pełny wariant | Zmiana |','|---|---:|---:|---:|---:|']
    for family,r in result.items():
        s=r['same_image_best_reference'];lines.append(f"| {family} | {s['n']} | {s['reference']:.6f} | {s['all_information']:.6f} | {s['delta']:+.6f} |")
    lines+=['','PS: ustalony fold 57 obrazów z kontynuacji użytej w #2675. TM: istniejące OOF plus zamrożony skan, 170 obrazów. SS: niezmieniony model uczony syntetycznie, 170 obrazów. Są to różne protokoły; nie tworzą wspólnego bezstronnego CV. Przewaga nad kompaktową kontrolą sama w sobie nie oznacza przewagi nad najlepszym systemem.','', '## Małe obszary','', '| Metoda | Obszar | N | Kontrola: IoU danej klasy | Pełny wariant: IoU danej klasy | Całkowite pominięcia: kontrola → pełny |','|---|---|---:|---:|---:|---|']
    for family,r in result.items():
        for group,key,miss in [('fg_le1','fg_iou','fg_miss'),('fg_1to5','fg_iou','fg_miss'),('clean_le1','clean_iou','clean_miss'),('clean_1to5','clean_iou','clean_miss')]:
            c=r['control'][group];f=r['all_information'][group]
            if c['n']:lines.append(f"| {family} | {group} | {c['n']} | {c[key]:.6f} | {f[key]:.6f} | {c[miss]} → {f[miss]} |")
    lines+=['','## Jak wykorzystano dane','',
      '- Wszystkie 510 rzeczywistych obrazów: każdy raz w walidacji i dwa razy w części treningowej trzech foldów; każdy kwalifikujący się obraz użyty w każdej z 20 epok.',
      '- Wszystkie 12 000 realizacji treningowych z istniejących pul synthetic_scaling: po 4000 na metodę, po dwa pełne przejścia oraz replay. 600 przykładów rozwojowych wykorzystano wyłącznie do pomiaru. Te pule nie oznaczają 12 000 unikalnych fotografii; źródła między metodami mogą się pokrywać.',
      '- RGB512, luminancja i resztkowy filtr przestrzenny; metoda wybiera model. Zamrożone cztery poziomy oryginalnego dekodera. Wszystkie bity znanej wiadomości i odpowiadające im podpisane logity; dla PS/TM dwa rozmiary wejścia.',
      '- Maska128: główna segmentacja. Maska512: dodatkowa segmentacja i granice. Odległość od granicy, centra składowych i pokrycie: zadania pomocnicze. full_watermark: pomocnicza klasyfikacja reżimu.',
      '- Dokładny syntetyczny czysty odpowiednik: energia różnicy, czyste negatywy i porównanie par. Dla rzeczywistych zdjęć czysty odpowiednik jest niedostępny, więc tych strat nie obliczamy.',
      '- Identyfikatory, hashe, parametry generatora i podziały: kontrola pochodzenia i kompletności. Historycznych, nakładających się wariantów nie liczymy jako nowych niezależnych zdjęć.',
      '- Maski, full_watermark i czyste odpowiedniki nie są wejściami przy predykcji. Zbioru testowego nie używano do uczenia ani strojenia.',
      '', '## RTX 3070 i protokół','',
      'Microbatch 2, efektywna partia 8, FP32, AdamW; oba warianty mają te same źródła, kolejność, seed i liczbę kroków. Cechy oryginalnego dekodera obliczono wcześniej na aktualnych pikselach. Nowy model jest kompaktowym dekoderem, a nie pełnym fine-tuningiem dużego modelu #2675.','',
      '| Metoda | Wariant | Szczyt pamięci tensorów GPU, GB | Trening i ocena trzech foldów, s |','|---|---|---:|---:|']
    for h in hardware:lines.append(f"| {h['family']} | {h['arm']} | {h['peak_gpu_GB']:.3f} | {h['fold_train_eval_seconds']:.1f} |")
    lines+=['','Czas foldów nie zawiera przygotowania danych ani treningu syntetycznego. Pełne czasy etapów, konfiguracje, kompletność użycia danych i pomiar uruchomieniowy znajdują się w results. Pamięć tensorów PyTorch nie jest całkowitą pamięcią pokazywaną przez nvidia-smi.','', '## Ograniczenia wniosku','',
      'Eksperyment testuje jeden konkretny sposób wspólnego wykorzystania danych, architekturę i budżet. Wynik ujemny nie dowodzi, że dodatkowe informacje są bezużyteczne. Wynik dodatni nie wskazuje, który składnik odpowiada za poprawę; wymagałoby to osobnych ablacji. Nie prognozujemy publicznego wyniku przez dodanie lokalnego przyrostu do 0,6440.','', '## Artefakty','',
      '[Pomiar porównawczy](results/measurements.json) · [Zmiany na każdym obrazie](results/per_image_comparison.csv) · [Pełny wynik](results/comparison.json)','']
    lines+=['## Odtworzenie danych syntetycznych','','| Metoda | Zgodne źródło, wiadomość i maska | Inny hash RGB niż historyczny |','|---|---:|---:|']
    for family in FAMILIES:
        lines.append(f"| {family} | 4200/4200 | {usage[family]['preparation']['historical_rgb_mismatches']} |")
    lines+=['','RGB wygenerowano ponownie oryginalnymi wrapperami watermarków; odnotowano różnice hashy względem historycznych realizacji. Nie deklarujemy identyczności pikselowej z dawnymi cache. Wszystkie cechy dekodera policzono od nowa z aktualnych pikseli, a oba porównywane warianty dostają dokładnie ten sam odtworzony zbiór.','']
    for family in FAMILIES:lines.append(f'![{family}: trzy największe regresje i trzy największe poprawy](results/{family}_example_comparison.png)\n')
    if (OUT/'auxiliary_diagnostics.json').exists():
        aux=read(OUT/'auxiliary_diagnostics.json')['families']
        lines+=['## Diagnostyka zadań pomocniczych','','Pomiary po treningu; bez wyboru lepszej głowicy, checkpointu lub progu na ich podstawie. Główny wynik pozostaje z ustalonej głowicy128.','','| Metoda | Główna maska128: OOF | Pomocnicza512 po majority: OOF | Trafność full_watermark | Błąd pokrycia |','|---|---:|---:|---:|---:|']
        for family in FAMILIES:
            a=aux[family]['real_oof'];lines.append(f"| {family} | {a['main']['all']['miou']:.6f} | {a['aux512']['all']['miou']:.6f} | {a['regime_correct']:.4f} | {a['coverage_abs_error']:.4f} |")
        lines+=['','Pełna diagnostyka energii resztkowej względem predykcji zerowej i lokalnej głowicy: [JSON](results/auxiliary_diagnostics.json). Zadania mogą nauczyć się własnych celów bez poprawy głównej segmentacji.','']
    first_cache=OUT/'data/pixelseal/complete.json'
    pipeline_seconds=(OUT/'runs/stegastamp/completed.json').stat().st_mtime-(first_cache.stat().st_mtime-usage['pixelseal']['preparation']['seconds'])
    lines+=['',f"Czas całego pipeline od rozpoczęcia przygotowania PixelSeal do końca treningu StegaStamp: {pipeline_seconds/60:.1f} min. Etapy częściowo nakładały się. Suma czasów porównań modelowych: {sum(r['seconds'] for r in result.values())/60:.1f} min.",'']
    save(OUT/'comparison.json',{'families':result,'overall':overall,'hardware':hardware,'pipeline_seconds':pipeline_seconds,'protocol':'matched 3-fold development CV, seed42, threshold0.5','official_images':510,'synthetic_training_realizations':12000,'synthetic_development_realizations':600})
    save(OUT/'measurements.json',measurements)
    with (OUT/'per_image_comparison.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:{'delta':v['delta'],'same_images_vs_best':v['same_image_best_reference']} for k,v in result.items()},indent=2))
if __name__=='__main__':report()
