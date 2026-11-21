"""Build Polish research report from measured artifacts only."""
from common import *
import pandas as pd

def table(rows,headers):
    def cell(x):
        if isinstance(x,(float,np.floating)):return '—' if not np.isfinite(x) else f'{x:.4f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(cell(x) for x in row)+' |' for row in rows])

def main():
    s=pd.read_csv(OUT/'summary_metrics.csv');primary=s[(s.watermark_method!='stegastamp')|((s.variant=='stable')&(s.seed==42))]
    size=pd.read_csv(OUT/'size_stratified_metrics.csv');size=size[(size.watermark_method!='stegastamp')|((size.variant=='stable')&(size.seed==42))]
    cal=pd.read_csv(OUT/'calibration_results.csv');probes=pd.read_csv(OUT/'feature_probe_results.csv');controls=pd.read_csv(OUT/'spatial_controls.csv')
    rank=pd.read_csv(OUT/'ranking_stratified_metrics.csv');alignment=json.loads((OUT/'alignment_probe_confirmation.json').read_text())
    tiny=json.loads((OUT/'tiny_overfit.json').read_text());geo=pd.read_csv(OUT/'geometry_stratified_metrics.csv')
    report=['''# Dlaczego modele rozpoznają obecność watermarku lepiej niż jego położenie

## Executive summary — najważniejsze ustalenia

1. **Wynik około 0,45–0,50 nie dowodzi dobrej lokalizacji.** Pusta maska na całym zbiorze daje tylko około 0,25, ale prosta decyzja całym obrazem „czysty albo watermarked”, wyprowadzona ze średniej probabilities, daje 0,4452 / 0,4491 / 0,4310 dla PixelSeal / TrustMark / stabilnego StegaStamp. Właściwe mapy dają 0,5078 / 0,4958 / 0,4562. Duża część wyniku pochodzi więc z rozpoznawania dominującej klasy obrazu. Wszystkie trzy modele słabo lokalizują małe watermarki i czyste dziury w dużych obszarach.

2. **StegaStamp ma rzeczywisty problem współrzędnych, ale nie naprawi go proste przesunięcie gotowych masek.** Aktywny STN przekształca obraz przed wszystkimi czterema pobieranymi cechami. Obecny trening porównuje je z GT w oryginalnym układzie. Sama odwrotna transformacja probabilities nie pomaga. Natomiast ponownie uczona sonda liniowa na wyrównanych cechach poprawia mIoU 0,4038 → 0,4235, ROC-AUC 0,6518 → 0,6709; poprawa mIoU we wszystkich 3 foldach. To dowód na użyteczność korekty w tej sondzie, nie obietnica wzrostu gotowej głowicy.

3. **Nie ma podstaw, żeby jako pierwszy ruch przebudowywać model na większy FPN.** Odtworzenie samych masek GT przez siatkę 32×32 daje około 0,95 mIoU, daleko powyżej modeli; nie jest to ścisły limit, ale wskazuje, że rozdzielczość sama nie tłumaczy całości strat. Wyższa rozdzielczość wczesnych cech nie pomogła sondom. Istniejąca głowica dopasowuje pięć obrazów do 0,963–0,989 treningowego mIoU. Problem dotyczy głównie lokalności sygnału, generalizacji i geometrii, nie podstawowej zdolności optymalizacji.

**Najbardziej uzasadniony następny eksperyment:** wytrenować od początku tę samą głowicę StegaStamp na cechach odwróconych przez rzeczywisty STN, z niezmienionym stable nested wyborem epoki/progu. Porównać z raw na identycznych foldach i seedach; bez przebudowy dekodera i bez submission.

### Najbardziej prawdopodobne ograniczenie każdej rodziny

| Metoda | Wniosek z wielu diagnostyk |
|---|---|
| PixelSeal | Najsilniejszy sygnał lokalny z trzech modeli, szczególnie dla średnich obszarów. Najmniejsze obiekty i wąskie pasy często giną; głębokie cechy oraz GRN niosą silny kontekst globalny. Użyteczny jest trzeci poziom, nie tylko najgłębszy. |
| TrustMark | Globalny odczyt i rozpoznanie dominującej klasy działają; małe regiony są często całkiem pomijane. Przy GT ≤5% AUC macro wynosi 0,435, FG IoU 0,0118. Nie wygląda to na sam brak ostrej granicy ani uniwersalnie zły próg. |
| StegaStamp | Łączy niezgodność geometrii STN, słabą lokalną separację małych regionów, globalną decyzję o obecności i przeszacowanie obszaru. Stabilizacja treningu pomaga głównie decyzji obszarowej; nie zapewnia dokładnej lokalizacji. |

## Zakres i wiarygodność

Wykonano audyt 510 unikalnych obrazów: oryginalne 3-fold OOF wszystkich metod oraz stabilny StegaStamp dla seedów 42/7/123. Plik per-image ma 1020 rekordów, ponieważ część obrazów oceniana jest w kilku wariantach; **to nadal 510 unikalnych obrazów**, nie 1020 niezależnych próbek.

W tabelach głównych: PixelSeal i TrustMark original seed42; StegaStamp stable seed42. Nie nazywamy tego identycznym protokołem treningu trzech rodzin. Oryginalny StegaStamp jest raportowany osobno. Publiczne 0,5123 nie jest wynikiem tego audytu i nie jest bezpośrednio porównywane z OOF.

Odtworzono wszystkie 18 zapisanych checkpointów i probabilities z tolerancją 1e−7. Wynik oficjalny pochodzi z funkcji startera, ze szczególną obsługą pustej unii. Dane testowe, predykcje konkursowe, dotychczasowe checkpointy i splity nie zostały zmienione. Wszystkie nowe modele to wyłącznie sondy i test tiny-set w osobnym katalogu.

Bootstrap 95%: 2000 losowań obrazów, parowany dla różnic, seed 20261121. Przedziały nie uwzględniają powtórnego treningu ani wielokrotnego wykorzystania tego zbioru do rozwoju. Screening sond na foldzie 1 jest eksploracyjny. Potwierdzenie korekty STN obejmuje także fold screeningowy.

[Galeria wszystkich 510 obrazów OOF z filtrem metod i przypadków](results/examples/index.html). [Pełne metryki per obraz](results/per_image_metrics.csv). [Podsumowanie](results/summary_metrics.csv).

## Architecture findings — faktyczny pipeline

Potwierdzono cztery poziomy cech i prostą głowicę fusion 32×32, zgodną z kodem. Istotne uzupełnienie: większe mapy są redukowane adaptive-average-pooling **przed cache i przed projekcją 1×1**, a nie dopiero przy łączeniu. PixelSeal native 64/32/16/8, TrustMark 62/31/16/8, StegaStamp 200/100/25/13; kanały odpowiednio 96/192/384/768, 256/512/1024/2048, 32/64/128/128.

PS/TM: RGB float, resize z antialias, normalizacja do [−1,1], eval i zamrożone parametry. Bufory BatchNorm nie zmieniły się. StegaStamp: NHWC 400, aktywny STN i centrowanie w grafie. Nie stwierdzono pomyłki nazw, BGR, transpozycji targetu ani odwrócenia maski. Wszystkie 510 masek 512 po oficjalnym downsamplingu są identyczne z targetem 128. Obrazy nie mają identycznych hashy; brak danych do pełnego audytu wspólnego źródła podobnych obrazów.

[Szczegóły warstw, wejść, pól recepcyjnych i checkpointów](architecture_audit.md). [Audyt preprocessingów i definicji metryk](preprocessing_audit.md).

## Spatial alignment — co naprawdę robi STN

Macierze zmierzono na wszystkich 170 obrazach StegaStamp. Średnie skale singular values 1,0735 i 1,0406; średni obrót −0,209°, SD 1,007°. Maksymalne przemieszczenie narożnika wynosi średnio około 31,9 px na obrazie 512. Transformacje nie są tożsamościowe.

Zrekonstruowano sampler ONNX, w tym nietypową stałą 398 zamiast automatycznie założonego 399/400; błąd odtworzenia obrazu <2,9e−5. Korekta cech uwzględnia środki convolutions i binów adaptive pool. Testy affine i konwencji środków przeszły.

| Kontrola | Wynik |
|---|---|
| Odwrócenie STN na probabilities stable, ta sama głowica | 0,4562 → 0,4555 |
| Odwrócenie STN na cechach, ta sama głowica | 0,4562 → 0,4259 |
| Ponowny trening identycznej sondy liniowej raw/aligned, 3 foldy | 0,4038 → 0,4235; delta +0,0198, CI [0,0161; 0,0233] |
| Macro AUC sondy raw/aligned | 0,6518 → 0,6709; CI delty [0,0109; 0,0271] |
| Macro AP sondy raw/aligned | 0,5617 → 0,5701 |

To **mierzalny składnik problemu**, nie dowód, że STN jest jedynym albo największym ograniczeniem. Głowica uczona na raw nie toleruje podmiany reprezentacji. Ponadto wyrównanie interpoluje cechy; brak osobnej kontroli samego wygładzenia ogranicza atrybucję przyczynową.

Wspólne małe przesunięcia probabilities nie dają znaczącej poprawy: najlepsze StegaStamp +0,00019. Flipy pogarszają. Per-image najlepsze korekty są oznaczone jako oracle. Translation-equivariance na 6 obrazach/metodę: korelacja mapy OOF po przesunięciu RGB o 16 px wynosi 0,910 / 0,860 / 0,438. StegaStamp jest wyraźnie mniej stabilny, ale test nie rozdziela STN od kontekstu i aliasingu.

[Pełny audyt STN](stn_alignment_audit.md). [Przykład RGB / STN / siatka](results/figures/stn_0.png). [Potwierdzenie na trzech foldach](results/alignment_probe_confirmation.json).

## Localization diagnostics — czy ranking pikseli działa?
''']
    report.append(table([[r.watermark_method,r.official_miou_mean,f'[{r.official_miou_ci_low:.4f}; {r.official_miou_ci_high:.4f}]',r.iou_watermark_mean,r.iou_clean_mean,r.roc_auc_mean,r.ap_mean,r.top_k_precision_mean] for r in primary.itertuples()],['Metoda','mIoU','95% CI','FG IoU','Clean IoU','AUC macro','AP macro','Top-K oracle']))
    report.append('''
Każdy wiersz ma 170 obrazów. Top-K korzysta z prawdziwej powierzchni GT, więc nie jest deployable. AUC/AP macro liczone per obraz, nie po wspólnej puli. W tym zbiorze nie ma obrazów jednorodnych; implementacja zachowuje NaN dla takich przypadków.

### Dlaczego pooled AUC bywa mylący
''')
    report.append(table([[r.watermark_method,r.pooled_roc_auc,r.roc_auc_mean,r.pooled_ap,r.ap_mean,r.prevalence,r.pooled_ap_prevalence_lift] for r in primary.itertuples()],['Metoda','AUC pooled','AUC macro','AP pooled','AP macro','Prevalence','Pooled AP / prevalence']))
    report.append('''
Pooled AUC około 0,93 może współistnieć z bardzo słabą lokalizacją wewnątrz obrazu: metoda nadaje wysokie probabilities obrazom prawie całym watermarked, a niskie obrazom prawie całym czystym. Macro oraz rozbicie po wielkości ujawniają ten problem. AP jest Average Precision; przewaga nad prevalence nie oznacza jeszcze przydatnej maski przy bardzo małym K.

### Pusta maska i kontrola bez lokalizacji
''')
    report.append(table([[r.watermark_method,r.empty_baseline_miou_mean,r.delta_vs_empty_mean,r.delta_vs_empty_median,r.delta_positive_fraction,r.delta_negative_fraction] for r in primary.itertuples()],['Metoda','Empty mIoU','Średnia delta','Mediana delta','Odsetek lepszych','Odsetek gorszych']))
    report.append(table([[r.method,r.control,r.miou,r.model_minus_control] for r in controls.itertuples() if r.control in ['image_mean_constant','independent_pixel_shuffle_5_repeats','oracle_empty_or_full']],['Metoda','Kontrola','mIoU kontroli','Model minus kontrola']))
    report.append('''
`image_mean_constant` ma jedną decyzję na obraz i nie wykorzystuje GT: porównuje średnią probabilities z zapisanym progiem. `oracle_empty_or_full` używa GT do wyboru całej klasy, więc jest wyłącznie diagnostyką. Shuffle zachowuje rozkład probabilities i powierzchnię predykcji, niszczy współrzędne; wynik to średnia 5 deterministycznych powtórzeń. Spadki po shuffle potwierdzają realny, ale ograniczony sygnał przestrzenny. Nie można utożsamić całej przewagi nad empty z lokalizacją.

### Separacja i Top-K
''')
    report.append(table([[r.watermark_method,r.probability_separation_mean,r.probability_separation_median,r.logit_separation_mean,r.effect_size_mean] for r in primary.itertuples()],['Metoda','Δp mean','Δp median','Δlogit mean','Effect size mean']))
    selected=rank[(rank.stratum=='small_le5pct')&(rank.seed==42)&((rank.method!='stegastamp')|(rank.variant=='stable'))]
    report.append(table([[r.method,r.n,r.gt_foreground_fraction,r.top_k_precision,r.roc_auc,r.ap,r.probability_separation] for r in selected.itertuples()],['GT ≤5%','n','Random Top-K baseline','Top-K oracle','AUC','AP','Δp inside−outside']))
    report.append('''
Dla małych StegaStamp nawet oracle-area wybór trafia tylko około 2,3% wybranych pikseli przy losowym odniesieniu około 1,9%. To nie wygląda na sam zły próg. Średnia separacja jest lekko ujemna. Dla PixelSeal sygnał jest lepszy, ale Top-K 6,1% nadal jest bardzo daleki od dobrej maski. TrustMark ma niskie macro AUC przy kilku lepszych hotspotach; AP i AUC opisują różne części rankingu.

## Geometry and resolution — rozmiar, dziury, granice
''')
    report.append(table([[r.watermark_method,r.size_bin,r.n,r.official_miou,r.iou_watermark,r.roc_auc,r.ap,r.delta_vs_empty,r.prediction_empty] for r in size.itertuples() if r.n],['Metoda','GT area','n','mIoU','FG IoU','AUC','AP','Δ empty','Empty rate']))
    report.append('''
Biny TrustMark 0–1% (n=9) i StegaStamp 20–50% (n=7) są szczególnie niepewne. Wszystkie trzy rodziny mają po 85 obrazów z GT >50%; tam średnie pokrycie GT to około 90%. FG IoU ~0,9 nie dowodzi wykrywania czystych otworów: dla StegaStamp w tej grupie Clean IoU wynosi zaledwie **0,0478**, a predykcja pokrywa średnio 96,9% obrazu.

StegaStamp przy GT ≤5% przewiduje średnio 16,5% foreground wobec GT 1,9%; delta do empty wynosi −0,0719. TrustMark w tej grupie zwykle pomija obiekt: FG IoU 0,0118, delta −0,0021. PixelSeal jest bliski empty średnio, choć ma część poprawnych lokalizacji.

Wąskie/wyciągnięte obszary (bbox aspect ≥4 lub ≤0,25): PixelSeal n=19, FG IoU 0,0570; StegaStamp n=22, FG IoU 0,0350 i AUC 0,4264. TrustMark n=7 — za mało do mocnej oceny. Kształt nie identyfikuje rodzaju edycji.

### Granice i komponenty
''')
    report.append(table([[r.watermark_method,r.boundary_f1_1_mean,r.boundary_f1_2_mean,r.boundary_f1_4_mean,r.boundary_internal_f1_2_mean,r.predicted_components_mean,r.largest_component_fraction_mean,r.gt_component_overlap_mean] for r in primary.itertuples()],['Metoda','BF1@1','BF1@2','BF1@4','Internal BF1@2','Pred components','Largest share','GT components hit']))
    report.append('''
**Ramka obrazu sztucznie ułatwia boundary score pełnych masek.** Po jej wyłączeniu Internal BF1@2 spada do 0,0765 / 0,0632 / 0,0496. Nie są to wyłącznie lekko rozmyte, poprawnie znalezione krawędzie; często brakuje właściwej granicy wewnętrznej. Dla pustych granic stosujemy jawne konwencje opisane w preprocessing_audit. Bbox IoU, centroid, komponenty, dziury, grubości i area ratio są w CSV. Porównania geometrii są skorelowane z powierzchnią — nie przypisujemy im niezależnej przyczynowości.

### Czy 32×32 jest głównym wąskim gardłem?
''')
    report.append(table([[r.watermark_method,r.gt_coarse32_reconstruction_miou_mean,r.gt_coarse32_reconstruction_fg_iou_mean] for r in primary.itertuples()],['Metoda','GT→avg32→bilinear128 mIoU','FG IoU']))
    report.append('''
To diagnostyka **z dostępem do GT**, nie model i nie matematyczny upper bound głowicy. Pokazuje, że sama siatka 32 potrafi reprezentować znacznie lepsze maski niż obecne predykcje. Najmniejsze obszary rzeczywiście tracą szczegóły: przy GT ≤1% FG IoU rekonstrukcji to około 0,593 / 0,808 / 0,783. Rozdzielczość może ograniczać końcową dokładność, ale nie tłumaczy niemal losowego rankingu małych watermarków.

[Geometria](results/geometry_stratified_metrics.csv), [metryki po wielkości](results/size_stratified_metrics.csv), [wykres PixelSeal](results/figures/pixelseal_diagnostics.png), [TrustMark](results/figures/trustmark_diagnostics.png), [StegaStamp](results/figures/stegastamp_diagnostics.png).

## Calibration — ile można odzyskać progami?
''')
    caltable=[]
    for r in primary.itertuples():
        z=cal[(cal.watermark_method==r.watermark_method)&(cal.variant==r.variant)&(cal.seed==r.seed)]
        global_score=float(np.average(z.global_resubstitution_miou,weights=z.n))
        caltable.append([r.watermark_method,r.official_miou_mean,z.global_resubstitution_threshold.iloc[0],global_score,r.oracle_threshold_miou_mean,r.oracle_threshold_gap_mean,r.brier_mean,r.log_loss_mean,r.ece])
    report.append(table(caltable,['Metoda','Operational','Global threshold*','Global mIoU*','Oracle per-image*','Oracle gap','Brier','BCE','ECE']))
    report.append('''
Gwiazdka: optymistyczna diagnostyka, nie oszacowanie poprawy wdrożeniowej. Globalny próg wybrany na OOF daje dla stable StegaStamp około +0,0155, znacznie mniej niż oracle +0,0634. PixelSeal odzyskuje około +0,0048. TrustMark jeden globalny próg jest gorszy od oddzielnych progów foldów.

Progi najlepszego globalnego wyniku na poszczególnych foldach są niestabilne: PixelSeal 0,64/0,93/0,62; TrustMark 0,73/0,48/0,89; original StegaStamp 0,52/0,75/0,32. Stabilny StegaStamp seed42: 0,69/0,89/0,89, ale optimum całego OOF zmienia się między seedami: 0,87 / 0,79 / 0,66. Nie należy przenosić bez testu jednego progu między checkpointami.

Cross-OOF calibration zapisano jawnie jako diagnostykę z overlapem treningowym, nie jako bezstronny nested wynik. Nie zastosowano nowej kalibracji do testu. Temperature scaling/isotonic nie były trenowane. Wyższy próg może ograniczyć nadmierny obszar StegaStamp, ale nie stworzy brakującego rankingu małych obiektów.

### Seed sensitivity
''')
    report.append(table([[r.variant,r.seed,r.official_miou_mean,r.roc_auc_mean,r.ap_mean,r.predicted_foreground_fraction_mean] for r in s[s.watermark_method=='stegastamp'].itertuples()],['StegaStamp wariant','Seed','OOF mIoU','AUC macro','AP macro','Pred area']))
    report.append('''
Średnia OOF stable po trzech seedach to około 0,46094; to nie dokładnie ta sama agregacja co średnia dziewięciu fold-seedów. Rankingi macro zmieniają się mniej niż mIoU original→stable, co pasuje do wcześniejszej diagnozy czasu treningu i area bias.

## Feature quality — skąd pochodzi lokalna informacja?

28 lekkich eksperymentów: cztery osobne poziomy, early, late, all, przestrzennie shuffled all, wyższa rozdzielczość early oraz aligned StegaStamp. Każda sonda to liniowe Conv1×1 na standaryzowanych kanałach; statystyki wyłącznie z części treningowej. BCE, AdamW 0,003, 30 epok, batch8, seed42. Próg z [0,3;0,4;0,5;0,6;0,7] na zapisanym inner holdout, następnie świeży fit outer-train. Screening: pierwszy oryginalny fold, n=57/metodę. Różne liczby kanałów oznaczają różną liczbę parametrów; nie jest to porównanie identycznej capacity między poziomami.
''')
    report.append(table([[r.method,r.variant,r.parameters,r.miou,r.roc_auc_macro,r.ap_macro] for r in probes.itertuples()],['Metoda','Sonda','Parametry','mIoU','AUC macro','AP macro']))
    fs=pd.read_csv(OUT/'cached_feature_statistics.csv')
    report.append('### Zakresy i zmienność cech w cache — wszystkie 170 obrazów/metodę')
    report.append(table([[r.method,r.layer,r.min,r.max,r.mean,r.std,r.mean_spatial_variance,r.zero_fraction] for r in fs.itertuples()],['Metoda','Poziom','Min','Max','Mean','SD','Spatial variance','Zero fraction']))
    report.append('Wszystkie wartości są skończone; żaden cały poziom nie jest stały lub całkowicie zerowy. ReLU daje dużo zer, np. około 71% w ostatnim StegaStamp, ale pozostałe aktywacje mają mierzalną zmienność. TrustMark layer4 ma 12,7% par obraz–kanał o niemal zerowej wariancji przestrzennej. Duże moduły aktywacji PixelSeal nie są same w sobie dowodem niestabilności; nie zaobserwowano NaN/Inf. Native statystyki na 6 obrazach/metodę i pełne statystyki cache zapisano osobno.')
    report.append('''
PixelSeal: trzeci poziom (stage2, native16×16) jest najlepszy według mIoU; najgłębszy etap traci ranking. TrustMark: layer2 daje lepszy AUC niż layer3, ale layer3 lepszy mIoU — wybór warstwy według samego mIoU pomija kalibrację. StegaStamp: najgłębsze mapy są wyraźnie bardziej informacyjne niż early. Wyższa rozdzielczość early nie pomaga żadnej rodzinie w sposób przekonujący, a StegaStamp pogarsza wynik.

Shuffle obniża AUC do okolic 0,5, mimo że mIoU pozostaje relatywnie wysokie. To potwierdza obecność realnej informacji przestrzennej w poprawnie zestawionych cechach, lecz także silną informację globalną. Sondy nie dowodzą, że w cechach nie ma żadnego sygnału nieliniowego — są testem liniowej dostępności. Wyższy cache StegaStamp ma 128×128 zamiast pełnego native200×200 pierwszego poziomu; nie testowano całej rozdzielczości 200.

Kontrola aligned/raw sondy StegaStamp została dodatkowo potwierdzona na foldach 2 i 3. Nie wykonano pełnego CV wszystkich pozostałych sond ani selekcji najlepszego poziomu na leaderboardzie. [Sondy po wielkości i względem głowicy na tym samym foldzie](results/probe_stratified_metrics.csv).

### Czy model potrafi nauczyć się kilku obrazów?
''')
    report.append(table([[r['method'],r['n'],r['miou'],r['fg_iou']] for r in tiny],['Metoda','Obrazy','TRAIN mIoU po 300 epokach','TRAIN FG IoU']))
    report.append('''
Pięć obrazów/metodę wybrano deterministycznie z różnych kwantyli powierzchni. Ta sama głowica, loss i optimizer, stały próg 0,5. To **in-sample sanity check**, nie wynik walidacyjny i nie dowód, że reprezentacja generalizuje. Wszystkie rodziny potrafią dopasować wybrane maski. Wynik nie uzasadnia automatycznego zwiększania capacity.

## Wiadomość: globalny odczyt nie jest mapą lokalizacji

Ponownie uruchomiono oryginalne wrappery dekoderów na wszystkich 510 obrazach. Logity porównano bezpośrednio z raw message w kolejności dostarczonych bitów: PixelSeal256, TrustMark100 bez ECC, StegaStamp100. Trafność jest zgodna z wcześniejszym cache. Nie przekształcamy surowych bitów w tekst ani nie stosujemy korekcji błędów.

Spearman bit accuracy ↔ GT area wynosi 0,855 / 0,895 / 0,850. Bit accuracy ↔ FG IoU: 0,867 / 0,904 / 0,851. Natomiast bit accuracy ↔ oficjalne mIoU: −0,034 / 0,063 / 0,223. Silna korelacja z FG IoU jest w dużej mierze powiązana z powierzchnią: duże watermarki łatwiej odczytać i łatwiej uzyskać wysokie FG IoU przewidując prawie wszystko. Zapisano także korelacje w binach area, confidence, mean signed log-likelihood i scatterplots; małe biny oraz stała trafność dają niestabilne lub NaN korelacje.

Nie powtarzano occlusion experiments: repo zawiera już `decoder_evidence`, w tym StegaStamp 4×4 occlusion i kontrolę złej wiadomości. Tamte wyniki używają innych wariantów niż główny audyt; nie mieszamy ich z matched porównaniem. Nie ma podstaw, żeby sam wysoki globalny bit accuracy traktować jako sukces lokalizacji.

## Failure cases — reprezentatywne błędy

- **Całkowite minięcie małego regionu:** TrustMark `77d08ec17b71`, GT 0,4%, predykcja pusta, mIoU 0,498, FG IoU 0, AUC 0,047. Pozornie dobry mIoU jest samym tłem. [Panel](results/examples/trustmark_original_42_039.jpg).
- **Katastrofalny błąd powierzchni:** StegaStamp `53934df0429a`, GT około 0,6%, predykcja 99,7%, mIoU 0,004, AUC 0,371. Głowica rozpoznaje prawie cały obraz jako watermarked. [Panel](results/examples/stegastamp_stable_42_005.jpg).
- **Ranking częściowo dobry, ale obszar źle wyznaczony:** StegaStamp `1256bb953ed7`, AUC 0,848, AP 0,297, GT 9,5%, predykcja około 0,2%, FG IoU 0. Silny hotspot jest poza właściwym prostokątem, pozostały sygnał zbyt słaby dla progu. [Panel](results/examples/stegastamp_stable_42_034.jpg).
- **Wąski pasek pominięty mimo AUC >0,7:** PixelSeal `4003d769516e`, AUC 0,783, AP 0,058, GT 2,7%, predykcja pusta. AUC nie wystarcza do dobrego wyboru nielicznych dodatnich pikseli. [Panel](results/examples/pixelseal_original_42_018.jpg).
- **Duże maski z pominiętymi czystymi dziurami:** bardzo wysokie FG IoU i AP współistnieją z niskim Clean IoU oraz Internal BF1. Pełna maska może znaleźć się w „5 best FG”, dlatego galeria nie ogranicza się do tego rankingu i pokazuje obie klasy oraz delta do empty.
- **Przesunięcia:** nie ma wspólnego flipu/offsetu naprawiającego modele; StegaStamp ma zależne od obrazu transformacje i gorszą equivariance. Część błędów jest geometryczna, ale nie wolno przesuwać każdej walidacyjnej predykcji według jej GT jako sposobu inferencji.

Galeria obejmuje wszystkie 510 głównych OOF, a tagi wskazują po 5 best FG, worst FG, small, high-AUC/low-IoU, low-AUC i empty-competitive na metodę. Kategorie mogą się nakładać. To celowa ochrona przed pokazywaniem wyłącznie wygodnych przykładów.

## Next experiments — ranking działań

| Priorytet | Eksperyment | Dowody i potencjał | Złożoność / compute | Ryzyko overfittingu | Co potwierdzi / odrzuci hipotezę |
|---|---|---|---|---|---|
| 1 | Ta sama głowica StegaStamp, raw vs inverse-STN features, świeży trening ze stable nested selection | Najsilniejszy nowy dowód: poprawa identycznej sondy we wszystkich foldach, również AUC/AP. Pozwala usunąć konkretną niezgodność współrzędnych bez przebudowy architektury. | Niska zmiana kodu; umiarkowany koszt 3 inner folds × outer folds, istniejący cache transformacji | Umiarkowane: development folds już wielokrotnie używane; z góry zamrozić porównanie i seedy | Poprawa parowanego OOF AUC/AP, FG IoU małych regionów i internal boundary na kilku foldach/seedach; brak poprawy wskazuje, że głowica już kompensuje geometrię lub ranking jest głównym limitem. Dodać kontrolę resamplingu, by oddzielić geometrię od wygładzenia. |
| 2 | Kontrolowany mały baseline RGB-only oraz RGB+stałe filtry wysokoczęstotliwościowe/SRM, przy porównywalnym budżecie | Brakuje odpowiedzi, ile daje watermark pretraining. Małe regiony mają prawie losowy ranking; kontrola sprawdzi nową informację zamiast jedynie większej capacity. | Średnia implementacja, większy compute niż probe; screening jednego foldu, bez strojenia wielu architektur | Wysokie przy 170 obrazach: trzymać parametry/budżet i z góry ustalone kryteria | Poprawa AP/Top-K małych regionów ponad RGB i obecne cechy, utrzymana po shuffle-control i na innych foldach. Jeśli RGB dorównuje watermark features, nie przypisywać poprawy specjalistycznemu pretrainingowi. |
| 3 | Rozszerzona, wyłącznie nested kalibracja StegaStamp na inner predictions; ewentualnie prosty parametr obszaru dopiero po kontroli | Globalne progi odzyskują małą część wyniku, a area overprediction jest duże. To tania kontrola, lecz nie remedium na brak rankingu. | Niska; ponowne użycie inner predictions, jeśli zapisane, inaczej odtworzenie inner modeli | Niskie dla 1–2 parametrów; duże dla rozbudowanej area head bez nowych danych | Parowana poprawa względem raw przy zachowaniu small-region recall; odrzucić, gdy redukcja FP tylko zastępuje maski pustymi lub próg nie przenosi się między foldami/seedami. |

Nie podajemy wymyślonych spodziewanych wzrostów punktów. Pierwszy eksperyment ma najmocniejsze bezpośrednie poparcie tego audytu. RGB/SRM jest hipotezą wartą testu, nie potwierdzonym ulepszeniem. Dotychczasowe failed loss/decoder128/patch2×2 nie są tu powtarzane pod inną nazwą.

### Odpowiedzi na 10 pytań końcowych

1. **Czy StegaStamp ma problem spatial alignment?** Tak: cechy po STN są porównywane z oryginalnym GT; matched retraining probe po odwróceniu pomaga na 3 foldach. Nie jest to pełne wyjaśnienie słabości.
2. **Czy ekstraktor zawiera użyteczną lokalizację?** Tak, ograniczoną: AUC >0,5 i spadek po shuffle. Dla małych watermarków obecna głowica jest bliska losowej.
3. **Które warstwy?** StegaStamp późne 6/7, szczególnie 7; PixelSeal stage2; TrustMark layer2–3 zależnie od AUC vs mIoU. To diagnostyka liniowej dostępności, nie ostateczny ranking każdej możliwej głowicy.
4. **Czy fusion32 traci szczegóły?** Tak dla najmniejszych struktur, ale pomiar GT reconstruction i high-resolution probes przeczą tezie, że to główna przyczyna wszystkich błędów.
5. **Ile wyniku wyjaśnia czyste tło?** Sama pusta maska około 0,25 globalnie, ale około 0,49 przy małych watermarkach. Decyzja całym obrazem bez lokalizacji daje 0,43–0,45; szczególnie StegaStamp uzyskuje niewiele więcej dzięki mapie przestrzennej.
6. **Czy kalibracja istotnie pomoże?** Umiarkowanie możliwe dla StegaStamp, optymistycznie około +0,0155 z jednego globalnego progu na stable seed42. To nie wynik bezstronnej walidacji. Ranking małych obiektów pozostaje problemem.
7. **Czy FPN pomoże?** Nie rozstrzygnięto; obecne dane nie uzasadniają go jako pierwszego kroku. Early high-resolution features nie pokazały przewagi, a obszary często nie są wykrywane w ogóle. FPN ma sens dopiero przy pokazaniu lokalnego sygnału wymagającego rekonstrukcji.
8. **Czy RGB/SRM branch jest uzasadniony?** Jako kontrolowany eksperyment z lokalnym rankingiem, tak; jako gotowa rekomendacja architektoniczna, jeszcze nie. RGB-only i SRM nie zostały wytrenowane w tym audycie.
9. **Czy priorytetem powinny być syntetyczne dane?** Jeszcze nie. Tiny-fit vs OOF wskazuje problem generalizacji, ale nie odróżnia braku danych od nieadekwatnej lokalnej reprezentacji. Najpierw korekta współrzędnych i baseline RGB/SRM; potem ewentualny eksperyment z danymi o kontrolowanej różnorodności. Nie wygenerowano watermarkowych danych syntetycznych.
10. **Jeden następny eksperyment?** Existing StegaStamp Head + inverse-STN features, trening od zera i matched stable nested control. Nie podmiana cech pod stary checkpoint.

## Co wykonano i czego nie zmierzono

Wykonano Priority 0 w całości: kod, dane, GT, checkpointy, STN, wszystkie OOF metryki, area/geometry/boundaries, galerię. Priority 1: probes, shuffle-control, equivariance, gradient RF PS/TM, threshold/calibration diagnostics, tiny overfit. Dodatkowo wszystkie globalne message decodery i potwierdzenie matched alignment probe na 3 foldach.

Nie wykonano: RGB-only/SRM treningu, nowych occlusions (istnieją wcześniejsze), label-randomization training, full-head retraining z korektą STN, temperature/isotonic calibration, pełnego CV wszystkich 28 sond, nowych architektur ani generowania syntetycznych watermarków. CPU PyTorch 2.5.1, bez dostępnego CUDA w tym środowisku; wybrano priorytetowe diagnostyki. Sześć obrazów dla native feature stat/equivariance i trzy dla gradient RF to ograniczone próby, jawnie oddzielone od pełnych 170-image statystyk cache i 510-image OOF.

Pojedyncze PNG/JPG i wykresy sprawdzono wizualnie. Galerię otwarto również w Chrome headless i sprawdzono zrzut widoku 1440×1100: panele i licznik 510 obrazów są widoczne. Galeria ma sprawdzone linki/coverage; nie deklarujemy ręcznej oceny wszystkich 510 obrazów ani pełnego testu interakcji filtrów.

## Odtwarzalność i źródła

[Instrukcja uruchomienia](README.md), [manifest źródeł i wersji](results/provenance.json), [kontrole wykonania](results/execution_checks.json), [testy końcowe](results/verification.json). Liczby pochodzą z lokalnych artefaktów i uruchomionych skryptów, nie z wyników publikacji upstream.

Źródło metryki i wrapperów: lokalny starter z [oficjalnego zbioru konkursowego](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris). Implementacje do identyfikacji pochodzenia: [PixelSeal / VideoSeal](https://github.com/facebookresearch/videoseal), [TrustMark](https://github.com/adobe/trustmark), [StegaStamp](https://github.com/tancik/StegaStamp), [WAVES](https://github.com/umd-huang-lab/WAVES). Konkretne lokalne wersje i sumy zapisano w provenance; dokumentacja upstream nie jest traktowana jako dowód rzeczywistego execution path.
''')
    text='\n\n'.join(report)
    (HERE/'final_research_report.md').write_text(text,encoding='utf-8');(HERE/'REPORT.md').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
