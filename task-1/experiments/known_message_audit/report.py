"""Aggregate paired image-level diagnostics without changing selection rules."""
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def ci(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    rng=np.random.default_rng(20261122)
    return np.quantile(values[rng.integers(len(values),size=(5000,len(values)))].mean(1),[.025,.975]).tolist()


def avg(values):return float(np.mean(values))


def summarize_maps(rows,method,family):
    groups={'all':rows,'small':[r for r in rows if 0<r['fg_fraction']<=.05],
            'large':[r for r in rows if r['fg_fraction']>.5],
            'global_embed':[r for r in rows if r['full_watermark']],
            'local_embed':[r for r in rows if not r['full_watermark']]}
    results=[]
    for group,rs in groups.items():
        if not rs:continue
        for key in rows[0]['metrics']:
            entry=dict(method=method,family=family,group=group,variant=key,n=len(rs),
                       auc=avg([r['metrics'][key]['auc'] for r in rs]),
                       ap=avg([r['metrics'][key]['ap'] for r in rs]))
            if key+'_null' in rows[0]['metrics']:
                for metric in ('auc','ap'):
                    delta=[r['metrics'][key][metric]-r['metrics'][key+'_null'][metric] for r in rs]
                    entry[metric+'_delta_null']=avg(delta)
                    entry[metric+'_delta_null_ci95']=ci(delta)
            results.append(entry)
    return results


def main():
    maps=[];thresholds=[];checks={};counts={}
    for method in ('pixelseal','trustmark'):
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/method).glob('*.json'))]
        counts[method]=len(rows)
        if not rows:continue
        maps+=summarize_maps(rows,method,'extractor')
        for group,rs in [('all',rows),('small',[r for r in rows if r['fg_fraction']<=.05]),
                         ('large',[r for r in rows if r['fg_fraction']>.5])]:
            for variant in ('baseline','contrast_true','retain95_true','contrast_null1','retain95_null1','contrast_null2','retain95_null2','oracle'):
                vals=[r['keep_drop'][variant] for r in rs]
                deltas=[r['keep_drop'][variant]-r['keep_drop']['baseline'] for r in rs]
                thresholds.append(dict(method=method,group=group,variant=variant,n=len(rs),
                                       miou=avg(vals),delta=avg(deltas),delta_ci95=ci(deltas)))
        quadrature=[r['verification'] for r in rows if 'ig16_sum' in r['verification']]
        checks[method]=dict(projection_max_error=max(r['verification']['projection_max_error'] for r in rows),
                            mean_bit_accuracy=avg([r['keep_drop']['original_bit_accuracy'] for r in rows]),
                            ig4_completeness_mae=avg([abs(r['verification']['ig4_sum'][0]-r['verification']['score_change'][0]) for r in rows]),
                            ig16_subset_n=len(quadrature),
                            ig4_subset_completeness_mae=avg([abs(r['ig4_sum'][0]-r['score_change'][0]) for r in quadrature]),
                            ig16_subset_completeness_mae=avg([abs(r['ig16_sum'][0]-r['score_change'][0]) for r in quadrature]),
                            ig4_ig16_mean_correlation=avg([r['ig4_ig16_correlation'] for r in quadrature]))
    for method in ('pixelseal','trustmark','stegastamp'):
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'reembed'/method).glob('*.json'))]
        counts['reembed_'+method]=len(rows)
        if rows:maps+=summarize_maps(rows,method,'reembed')
    gates=[]
    for r in maps:
        if r['group']!='all' or 'auc_delta_null_ci95' not in r:continue
        small=next((s for s in maps if s['method']==r['method'] and s['variant']==r['variant'] and s['group']=='small'),None)
        passed=r['auc']>=.60 and r['auc_delta_null_ci95'][0]>0 and small is not None and small['ap_delta_null']>=0
        gates.append(dict(method=r['method'],variant=r['variant'],passed=passed))
    fusion=[];controls=[]
    for method in ('pixelseal','trustmark'):
        p=OUT/'fusion'/f'{method}_summary.json'
        if p.exists():fusion.append(json.loads(p.read_text(encoding='utf-8')))
    for method in ('pixelseal','trustmark','stegastamp'):
        p=OUT/f'reembed_controls_{method}.json'
        if p.exists():
            c=json.loads(p.read_text(encoding='utf-8'))
            controls.append(dict(method=method,n=c['n'],auc=avg([r['metrics']['ncc_true']['auc'] for r in c['rows']]),
                                 null_auc=avg([r['metrics']['ncc_wrong']['auc'] for r in c['rows']]),
                                 bit_accuracy=avg([r['full_marked_bit_accuracy'] for r in c['rows']])))
    summary=dict(counts=counts,maps=maps,thresholds=thresholds,checks=checks,gates=gates,fusion=fusion,controls=controls,
                 protocol='Repeatedly used real development images; frozen-model diagnostics; no test submission. Paired bootstrap by image, unadjusted multiple comparisons.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    for filename,rows in [('map_metrics.csv',maps),('threshold_metrics.csv',thresholds)]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# Znana wiadomość: pomiary nowych propozycji','',
           '**Nie potwierdzono zapowiadanych dużych zysków.** Poniższe wyniki rozdzielają sygnał wiadomości od poprawy gotowego segmentatora.',
           '', '[Plan i reprodukcja](README.md) · [Audyt źródeł i matematyki](RESEARCH.md)', '',
           '## Zakres i protokół', '',
           f'Wykonane liczby obrazów: `{counts}`. PS/TM: natywne ekstraktory i dwie skale, IG od rozmytego obrazu, Keep/Drop oraz pilot uczonej fuzji. Wszystkie trzy metody: re-embedding NCC. Żaden nowy model nie został zgłoszony.',
           '', 'Podstawą Keep/Drop są zapisane OOF ensemble trzech seedów: PixelSeal 4k/2500 i TrustMark 4k/5000. Każdy próg dobierany z wiadomości i logitów, bez użycia GT. Oryginalny próg jest jednym z kandydatów. Oracle używa GT i jest wyłącznie diagnostyką.',
           '', '## Mapy: macro ROC-AUC / Average Precision', '',
           'Macro oznacza średnią metryki liczonej osobno na każdym obrazie. AP zależy od powierzchni maski; przy braku sygnału porównujemy z null i raportujemy małe regiony osobno.', '',
           '| Metoda | Mapa | AUC | AP | AUC minus null [CI95] |',
           '|---|---|---:|---:|---|']
    display=['baseline_probability','dense256_cos','dense245_cos','dense512_cos','multiscale_cos','blur_ig','blur_ig_calibrated','ncc16','ncc32','ncc32_calibrated']
    for r in maps:
        if r['group']=='all' and r['variant'] in display:
            diff='—'
            if 'auc_delta_null' in r:
                lo,hi=r['auc_delta_null_ci95'];diff=f"{r['auc_delta_null']:+.4f} [{lo:+.4f}; {hi:+.4f}]"
            lines.append(f"| {r['method']} | {r['variant']} | {r['auc']:.4f} | {r['ap']:.4f} | {diff} |")
    lines+=['','Wszystkie mapy, podziały small/large i sposób osadzania: [map_metrics.csv](results/map_metrics.csv). Dodatnia różnica względem losowej wiadomości dowodzi sygnału zależnego od wiadomości, nie zysku po fuzji.',
            '', '## Keep/Drop: rzeczywista zmiana mIoU', '',
            '| Metoda | Kryterium | mIoU | Delta do bazy | CI95 delty |', '|---|---|---:|---:|---|']
    for r in thresholds:
        if r['group']=='all':
            lo,hi=r['delta_ci95']
            lines.append(f"| {r['method']} | {r['variant']} | {r['miou']:.5f} | {r['delta']:+.5f} | [{lo:+.5f}; {hi:+.5f}] |")
    lines+=['', '`contrast`: maksimum signed score keep minus drop. `retain95`: najmniejsza maska zachowująca >=95% przyrostu score względem rozmycia i zostawiająca <=5% poza maską; brak spełnienia warunku zachowuje próg bazowy. Null1/2 to osobne fałszywe wiadomości. Nie wybrano najlepszego progu na GT.',
            '', '### Małe znaki (GT <=5%)', '', '| Metoda | Kryterium | mIoU | Delta |', '|---|---|---:|---:|']
    for r in thresholds:
        if r['group']=='small' and r['variant'] in ('baseline','contrast_true','retain95_true'):
            lines.append(f"| {r['method']} | {r['variant']} | {r['miou']:.5f} | {r['delta']:+.5f} |")
    lines+=['', '## Czy doklejenie map pomaga segmentatorowi?', '',
            'Po sprawdzeniu sygnału uruchomiono małą korektę zamrożonego segmentatora seed42. Dwa ramiona mają identyczną architekturę, inicjalizację, kolejność danych i procedurę wyboru; różnią się poprawnością wiadomości. Każde ramię wybiera epokę i próg na inner-validation, potem jest trenowane od nowa na outer-train. Wyjściowa korekta zerowa odtwarza bazę. Dane outer-validation nie wybierają epoki ani progu.',
            '', '| Metoda | Fold | Baza | Fuzja true | Fuzja wrong | True model z wrong w inference | Bramka CV |',
            '|---|---:|---:|---:|---:|---:|---|']
    for f in fusion:
        for r in f['folds']:
            v=r['variants']
            lines.append(f"| {f['method']} | {r['fold']} | {r['baseline_miou']:.5f} | {v['real']['miou']:.5f} | {v['wrong']['miou']:.5f} | {v['real']['counterfactual_miou']:.5f} | {'TAK' if f['screen_passed'] else 'NIE'} |")
    lines+=['', 'Bramka: co najmniej +0,005 nad bazą i nad modelem uczonym z wrong na foldzie 1. Wyniki tego pilota nie stanowią pełnego CV ani pomiaru na niezależnym teście. Niespełnienie bramki zatrzymuje dalsze foldy. Odtworzenie wewnętrznego modelu bazowego sprawdzono względem zapisanej historii oryginalnego treningu.',
            '', '## Kontrola dodatnia re-embedding', '',
            'Po negatywnym teście realnych danych sprawdzono osiem deterministycznie wybranych coverów COCO na metodę. Oryginalne osadzenie na pełnym obrazie, centralna czysta dziura 256x256, ta sama wiadomość, rozdzielczość i faza. Osobno czysty cover i fałszywa wiadomość. To mała kontrola diagnostyczna po wynikach, nie nowy wynik konkursowy.',
            '', '| Metoda | N | AUC NCC32 true | AUC wrong | Trafność bitów na pełnym watermarku |', '|---|---:|---:|---:|---:|']
    for r in controls:
        lines.append(f"| {r['method']} | {r['n']} | {r['auc']:.4f} | {r['null_auc']:.4f} | {r['bit_accuracy']:.4f} |")
    lines+=['', '## Kontrole implementacji i ograniczenia', '',
            '- Gęste logity odtwarzają średni odczyt oryginalnego ekstraktora; sprawdzone na każdym obrazie. Rozdzielczość 512 celowo omija resize wrappera, więc jest zmianą skali ekstraktora.',
            '- `blur_ig` to dokładnie wzór z dostarczonej propozycji: **straight-line integrated gradients od rozmytej bazy**, nie oryginalny algorytm BlurIG ze ścieżką zmiennego sigma. RGB sumowane po kanałach; nie wykonano osobnej ścieżki wyłącznie luminancji.',
            '- Cztery prawe końce przedziałów nie zapewniają dokładnej kompletności. Test 16 kroków obejmuje osiem z góry ustalonych obrazów na rodzinę; pozostający błąd jest jawnie raportowany.',
            '- Odejmowanie średniej dwóch losowych wiadomości nazwano w plikach `calibrated` dla zgodności z propozycją. To szumna kontrola kontrfaktyczna, nie udowodnione usunięcie biasu tekstury. Dla liniowego signed score oczekiwanie po losowej wiadomości wynosi już zero.',
            '- Re-embedding wykorzystuje pełny obraz i oryginalne wrappery uint8. Brak zgodności fazy, cropa albo odpowiedzi zależnej od treści może pogarszać wynik. Negatywny test nie wyklucza metody z poprawnie odtworzoną geometrią.',
            '- Powtarzane dane rozwojowe, wiele map bez korekty wielokrotnych porównań. CI obejmują bootstrap sparowanych obrazów, nie ponowny trening ani nowe zdjęcia.',
            '', '```json',json.dumps(checks,indent=2),'```', '', '## Dalsze decyzje', '',
            'Mapy przechodzące ustaloną bramkę sygnału (AUC >=.60, dolny CI przewagi nad null >0, small AP nie gorsze od null):', '']
    passed=[r for r in gates if r['passed']]
    lines += [f"- {r['method']}: `{r['variant']}`" for r in passed] or ['- Brak.']
    lines += ['', '**Żaden wariant nie uzyskał w tych testach podstaw do wdrożenia.** Piloty fuzji nie przeszły bramki, a +0,00257 mIoU Keep/Drop dla PixelSeal ma CI95 obejmujący zero i lekko pogarsza małe znaki. Obietnica natychmiastowego +0,03–0,08 nie znajduje potwierdzenia.',
              '', '**Najbardziej konkretny dalszy kierunek: re-embedding z dopasowaniem geometrii dla StegaStamp.** NCC32 ma AUC 0,60714 na 85 realnych obrazach z globalnym osadzeniem i 0,54295 na 85 z lokalnym osadzeniem; przy pełnej zgodności geometrii na ośmiu syntetycznych kontrolach AUC wynosi 0,68798. To poszlaka, nie dowód, że sama korekta geometrii rozwiąże różnicę domen. Kolejny test powinien izolować skalę, położenie i zależność residualu od treści na sparowanych coverach, a dopiero potem sprawdzać fuzję.',
              '', 'Bramka sygnału uzasadniła opisany wyżej test małej fuzji. Sama zgodność wiadomości nie wystarcza do zastąpienia obecnego modelu. Nie należy wdrażać surowego maksimum keep-minus-drop, ani traktować wartości oracle jako wyniku osiągalnego bez etykiet.',
              '', 'Nie testowano tu pełnego odmrażania ekstraktora z masked bit loss, różniczkowalnej optymalizacji maski, segmentacji spójnych składowych ani konwersji ONNX StegaStamp do autograd. Wyniki nie obalają tych odrębnych hipotez.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'counts':counts,'keep_drop':[r for r in thresholds if r['group']=='all' and r['variant'] in ('baseline','retain95_true','contrast_true')],'passed':passed},indent=2))


if __name__=='__main__':main()
