# RGB / SRM: wyniki eksperymentu

**Status: pełny zaplanowany screening i potwierdzenie ukończone.**

A = zamrożony ekstraktor + dotychczasowy Head; B = RGB; C = RGB + stałe residuals SRM-style; D = ekstraktor + RGB/SRM. W StegaStamp A/D używają poprawionego inverse-STN. Wyniki są zmierzone na GPU, bez danych syntetycznych, testu konkursowego i submission.
## Odpowiedzi i rekomendacja

**Zachować dotychczasowe najlepsze rozwiązanie. D warto dalej badać tylko dla StegaStamp;
nie ma podstaw do zastąpienia nim wszystkich trzech modeli.** Ukończono 60 przebiegów:
54 dla A/D (3 metody × 3 foldy × 3 seedy) i 6 screeningowych B/C.

1. **Czy RGB-only przewyższa cechy ekstraktora? Nie w tym screeningu.** Na tym samym
   foldzie macro AP A→B wynosi PixelSeal 0.6066→0.5037, TrustMark 0.5364→0.4854,
   StegaStamp 0.6352→0.5873. RGB poprawia small AP StegaStamp 0.0318→0.0443,
   ale obniża jego small Top-K 0.0392→0.0307. Żaden B/C nie spełnił kryterium
   awansu. To dowód przewagi tych ekstraktorów nad tą konkretną lekką CNN;
   nie rozdziela wpływu pretrainingu od większej pojemności zamrożonych modeli.

2. **Czy SRM pomaga? Brak spójnego potwierdzenia.** W screeningu B→C small AP
   rośnie dla PixelSeal 0.0252→0.0354 i TrustMark 0.0330→0.0365, lecz spada dla
   StegaStamp 0.0443→0.0222. Macro AP C jest niższe od B we wszystkich metodach.
   C nie potwierdzano na innych foldach/seedach, więc nie przesądzamy o SRM ogólnie.

3. **Czy fuzja pomaga? Lokalnie dla StegaStamp, bez poprawy globalnego mIoU.**
   Po 3 foldach × 3 seedach D podnosi jego small AP 0.0425→0.0564 (+0.0139),
   Top-K 0.0312→0.0452 (+0.0140) i FG IoU 0.0153→0.0205. Small AP i Top-K
   poprawiają się w średniej OOF każdego z trzech seedów, ale tylko w dwóch
   z trzech foldów: fold3 AP 0.0314→0.0295, Top-K 0.0220→0.0123.
   Warunkowy CI95 delty small AP wynosi [0.0022; 0.0284]; dla Top-K
   [−0.0004; 0.0311], dla FG IoU [−0.0012; 0.0137]. To obiecujący, ograniczony
   sygnał. W PixelSeal small AP spada 0.0775→0.0716, w TrustMark 0.0716→0.0672;
   również small Top-K obu metod spada. TrustMark zyskuje globalne AUC
   0.6987→0.7153, ale traci small FG IoU i clean IoU dużych masek 0.1298→0.1071.

4. **Czy to rzeczywista lokalizacja? Zysk StegaStamp dotyczy rankingu, nie tylko
   dominującej klasy.** AP/Top-K nie zależą od wybranego progu; oba rosną,
   podczas gdy OOF mIoU spada 0.4682→0.4664. D przewyższa losowy ranking
   małych masek: AP 0.0564 vs 0.0196, Top-K 0.0452 vs 0.0190. Nadal są to
   słabe wyniki bezwzględne: small FG IoU zaledwie 0.0205, small AUC spada
   0.5203→0.5110, a small mIoU 0.4394 jest niższe niż 0.4905 pustej maski.
   Poprawa dotyczy najlepszych punktów rankingu; problem lokalizacji nie został rozwiązany.

5. **Następny krok: kontrolowana ablacja D dla StegaStamp.** Zachować obecną
   architekturę A dla PixelSeal/TrustMark oraz najlepsze wcześniejsze rozwiązanie.
   Jako następny eksperyment porównać D z identyczną fuzją, ale wyzerowanymi
   dziewięcioma kanałami SRM (ta sama liczba parametrów, inicjalizacja i budżet),
   na wszystkich tych samych foldach/seedach; główne kryteria to small AP i Top-K.
   Pozwoli to ustalić, czy sygnał D pochodzi z residuals, czy z gałęzi RGB.
   Nie zmieniać jednocześnie loss, selekcji epoki i dekodera. Szczególnie
   sprawdzić fold3, gdzie zysk się nie przenosi. Wdrażanie D teraz byłoby przedwczesne.

Wszystkie wnioski dotyczą rozwojowych foldów wielokrotnie wykorzystywanych w projekcie.
Pełne wyniki poniżej rozdzielają screening od OOF; nie wykorzystano leaderboardu.



## Porównanie

| Metoda | Wariant | Protokół | OOF mIoU | AUC macro | AP macro | AP ≤5% | Top-K ≤5% | FG IoU ≤5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixelseal | A | 3-fold x 3 seeds | 0.5114 | 0.7144 | 0.6310 | 0.0775 | 0.0606 | 0.0382 |
| pixelseal | B | screen fold1 seed42 | 0.2614 | 0.6030 | 0.5037 | 0.0252 | 0.0117 | 0.0156 |
| pixelseal | C | screen fold1 seed42 | 0.2478 | 0.5967 | 0.5002 | 0.0354 | 0.0250 | 0.0152 |
| pixelseal | D | 3-fold x 3 seeds | 0.5107 | 0.7119 | 0.6324 | 0.0716 | 0.0563 | 0.0370 |
| stegastamp | A | 3-fold x 3 seeds | 0.4682 | 0.6665 | 0.5717 | 0.0425 | 0.0312 | 0.0153 |
| stegastamp | B | screen fold1 seed42 | 0.2743 | 0.5077 | 0.5873 | 0.0443 | 0.0307 | 0.0240 |
| stegastamp | C | screen fold1 seed42 | 0.2491 | 0.5123 | 0.5818 | 0.0222 | 0.0021 | 0.0116 |
| stegastamp | D | 3-fold x 3 seeds | 0.4664 | 0.6708 | 0.5796 | 0.0564 | 0.0452 | 0.0205 |
| trustmark | A | 3-fold x 3 seeds | 0.5024 | 0.6987 | 0.6017 | 0.0716 | 0.0576 | 0.0303 |
| trustmark | B | screen fold1 seed42 | 0.2821 | 0.5238 | 0.4854 | 0.0330 | 0.0228 | 0.0016 |
| trustmark | C | screen fold1 seed42 | 0.2776 | 0.5470 | 0.4727 | 0.0365 | 0.0249 | 0.0226 |
| trustmark | D | 3-fold x 3 seeds | 0.5007 | 0.7153 | 0.6063 | 0.0672 | 0.0500 | 0.0227 |

## Screening: identyczny fold1 i seed42

| Metoda | Wariant | OOF mIoU | AUC macro | AP macro | AP ≤5% | Top-K ≤5% | FG IoU ≤5% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pixelseal | A | 0.5148 | 0.6914 | 0.6066 | 0.0373 | 0.0248 | 0.0068 |
| pixelseal | B | 0.2614 | 0.6030 | 0.5037 | 0.0252 | 0.0117 | 0.0156 |
| pixelseal | C | 0.2478 | 0.5967 | 0.5002 | 0.0354 | 0.0250 | 0.0152 |
| pixelseal | D | 0.5024 | 0.6724 | 0.5945 | 0.0176 | 0.0034 | 0.0087 |
| stegastamp | A | 0.4687 | 0.6399 | 0.6352 | 0.0318 | 0.0392 | 0.0077 |
| stegastamp | B | 0.2743 | 0.5077 | 0.5873 | 0.0443 | 0.0307 | 0.0240 |
| stegastamp | C | 0.2491 | 0.5123 | 0.5818 | 0.0222 | 0.0021 | 0.0116 |
| stegastamp | D | 0.4779 | 0.6768 | 0.6587 | 0.0722 | 0.0575 | 0.0182 |
| trustmark | A | 0.4597 | 0.6740 | 0.5364 | 0.0530 | 0.0475 | 0.0339 |
| trustmark | B | 0.2821 | 0.5238 | 0.4854 | 0.0330 | 0.0228 | 0.0016 |
| trustmark | C | 0.2776 | 0.5470 | 0.4727 | 0.0365 | 0.0249 | 0.0226 |
| trustmark | D | 0.4896 | 0.6864 | 0.5358 | 0.0421 | 0.0222 | 0.0173 |

Wiersze screeningowe mają tylko fold1/seed42 i nie są porównywalne z pełnym OOF. Wyniki pełne to średnia po obrazach OOF i trzech seedach, nie ensemble probabilities. Każda metoda ma 170 unikalnych obrazów; seedy nie zwiększają liczby niezależnych przykładów.

## Kontrole lokalizacji i dominującej klasy

| Metoda | Wariant | FG IoU | Clean IoU | Clean IoU GT>50% | Empty mIoU | Decyzja całym obrazem mIoU | Losowe AP | Losowe AP ≤5% | Losowe Top-K ≤5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixelseal | A | 0.5180 | 0.5048 | 0.1301 | 0.2530 | 0.4480 | 0.4942 | 0.0174 | 0.0168 |
| pixelseal | B | 0.4323 | 0.0904 | 0.0759 | 0.2643 | 0.2357 | 0.4717 | 0.0147 | 0.0141 |
| pixelseal | C | 0.4073 | 0.0884 | 0.0481 | 0.2643 | 0.2318 | 0.4717 | 0.0147 | 0.0141 |
| pixelseal | D | 0.5154 | 0.5060 | 0.1329 | 0.2530 | 0.4475 | 0.4942 | 0.0174 | 0.0168 |
| stegastamp | A | 0.4870 | 0.4494 | 0.0873 | 0.2540 | 0.4415 | 0.4923 | 0.0196 | 0.0190 |
| stegastamp | B | 0.4386 | 0.1100 | 0.0292 | 0.2221 | 0.2501 | 0.5561 | 0.0175 | 0.0169 |
| stegastamp | C | 0.0940 | 0.4042 | 0.0815 | 0.2221 | 0.2221 | 0.5561 | 0.0175 | 0.0169 |
| stegastamp | D | 0.4852 | 0.4476 | 0.0866 | 0.2540 | 0.4364 | 0.4923 | 0.0196 | 0.0190 |
| trustmark | A | 0.5011 | 0.5038 | 0.1298 | 0.2484 | 0.4485 | 0.5035 | 0.0245 | 0.0239 |
| trustmark | B | 0.0292 | 0.5349 | 0.0932 | 0.2724 | 0.2724 | 0.4555 | 0.0252 | 0.0247 |
| trustmark | C | 0.1035 | 0.4517 | 0.0969 | 0.2724 | 0.2724 | 0.4555 | 0.0252 | 0.0247 |
| trustmark | D | 0.5092 | 0.4921 | 0.1071 | 0.2484 | 0.4495 | 0.5035 | 0.0245 | 0.0239 |

AUC/AP są średnimi metryk liczonych oddzielnie dla każdego obrazu. Top-K zna prawdziwą liczbę dodatnich pikseli: jest diagnostyką oracle, nie maską deployable. Remisy Top-K są rozwiązywane przez oczekiwaną precyzję losowego wyboru wśród remisów. Losowy ranking ma AUC 0.5 i oczekiwane Top-K równe GT area; AP wykorzystuje dokładną wartość oczekiwaną dla losowej permutacji (bliską prevalence). Empty to faktyczna pusta maska z oficjalną obsługą pustej unii. Decyzja całym obrazem: pełna maska gdy średnia probability ≥ wybrany inner próg, inaczej pusta.

## Zmienność i budżet

| Metoda | Wariant | Parametry trenowane | Parametry zamrożone | Sekundy GPU jobs | mIoU SD seedów | mIoU SD foldów | Small AP SD seedów | Small Top-K SD seedów |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixelseal | A | 92705 | 33372545 | 28.0712 | 0.0016 | 0.0111 | 0.0029 | 0.0058 |
| pixelseal | B | 70281 | 0 | 7.3989 | — | — | — | — |
| pixelseal | C | 71577 | 0 | 12.2191 | — | — | — | — |
| pixelseal | D | 118041 | 33372545 | 56.6329 | 0.0032 | 0.0097 | 0.0044 | 0.0039 |
| stegastamp | A | 57889 | 52505482 | 65.7177 | 0.0129 | 0.0143 | 0.0103 | 0.0115 |
| stegastamp | B | 70281 | 0 | 11.7229 | — | — | — | — |
| stegastamp | C | 71577 | 0 | 8.1546 | — | — | — | — |
| stegastamp | D | 83225 | 52505482 | 192.4566 | 0.0040 | 0.0169 | 0.0044 | 0.0062 |
| trustmark | A | 169505 | 23712932 | 21.4359 | 0.0185 | 0.0176 | 0.0061 | 0.0103 |
| trustmark | B | 70281 | 0 | 5.8976 | — | — | — | — |
| trustmark | C | 71577 | 0 | 2.2179 | — | — | — | — |
| trustmark | D | 194841 | 23712932 | 56.2932 | 0.0052 | 0.0072 | 0.0064 | 0.0073 |

Budżet optymalizacji jest taki sam: batch8, maks.50 epok, patience8, AdamW lr0.001/decay0.0001, 0.5BCE+0.5Dice; epoka i próg wybierane identycznie wewnątrz train. Czasy obejmują wybór, refit, ocenę i zapis checkpointu; nie obejmują historycznej ekstrakcji cech. PS/TM używają dotychczasowego holdoutu80/20, StegaStamp trzech wewnętrznych foldów. Większy koszt SS nie jest dodatkowym budżetem tylko dla nowej architektury. GPU RTX3070 i GTX1080Ti, float32 bez TF32. Różne liczby parametrów i operacje GPU oznaczają, że to porównanie reprezentacji przy tym samym algorytmie treningu, nie ścisły capacity/FLOP match.

Parametry zamrożone obejmują cały ekstraktor/dekoder (StegaStamp także STN), bez enkodera. W CSV dodatkowo liczba parametrów backbone potrzebnego do czterech cech, bez końcowego odczytu wiadomości. Cache cech eliminuje koszt ekstraktora w pętli treningowej. Modele uczono na dwóch różnych kartach; brak ścisłej deterministyczności między sprzętem i CPU. Dlatego podstawą porównania jest nowy A na GPU, a nie historyczne wyniki CPU.

## Screening i potwierdzenie

| Metoda | Wariant | Awans |
| --- | --- | --- |
| pixelseal | B | False |
| pixelseal | C | False |
| trustmark | B | False |
| trustmark | C | False |
| stegastamp | B | False |
| stegastamp | C | False |

Kryterium zapisane przed treningiem: poprawa obu AP i Top-K małych masek względem A oraz macro AP ≥ A−0.01. Nie wybierano modelu na podstawie globalnego mIoU. D zawsze przechodzi pełne CV. Poniżej delty D−A na foldach2/3, poza screeningiem:

| Metoda | Metryka | Delta D−A | CI95 low | CI95 high |
| --- | --- | --- | --- | --- |
| pixelseal | miou | 0.0004 | -0.0045 | 0.0055 |
| pixelseal | ap | 0.0065 | -0.0007 | 0.0140 |
| pixelseal | auc | 0.0060 | -0.0078 | 0.0198 |
| pixelseal | topk | 0.0077 | -0.0013 | 0.0173 |
| pixelseal | fg_iou | -0.0050 | -0.0103 | -0.0002 |
| pixelseal | clean_iou | 0.0057 | -0.0014 | 0.0132 |
| stegastamp | miou | -0.0002 | -0.0072 | 0.0069 |
| stegastamp | ap | 0.0063 | -0.0019 | 0.0151 |
| stegastamp | auc | 0.0076 | -0.0068 | 0.0236 |
| stegastamp | topk | 0.0029 | -0.0062 | 0.0128 |
| stegastamp | fg_iou | 0.0063 | -0.0024 | 0.0154 |
| stegastamp | clean_iou | -0.0067 | -0.0165 | 0.0028 |
| trustmark | miou | -0.0081 | -0.0192 | 0.0017 |
| trustmark | ap | 0.0047 | -0.0075 | 0.0169 |
| trustmark | auc | 0.0225 | 0.0063 | 0.0397 |
| trustmark | topk | 0.0069 | -0.0061 | 0.0197 |
| trustmark | fg_iou | 0.0122 | 0.0017 | 0.0218 |
| trustmark | clean_iou | -0.0283 | -0.0456 | -0.0126 |

Bootstrap: 2000 parowanych losowań obrazów, po uśrednieniu różnic po seedach. Przedziały warunkowe nie obejmują niepewności treningu ani adaptacyjnego wykorzystania tego zbioru w poprzednich eksperymentach. SD foldów i seedów opisowe.

## Artefakty

[Tabela porównań](results/comparison.csv), [wszystkie obrazy i metryki](results/per_image_metrics.csv), [foldy](results/fold_metrics.csv), [seedy](results/seed_metrics.csv), [parowane różnice](results/paired_deltas.csv), [konfiguracja](results/config.json), [galeria seed42](results/gallery/index.html). Pełne float32 heatmaps i maski dla wszystkich seedów w plikach NPZ; ich PNG w results/heatmaps/.

[Weryfikacja](results/verification.json): wszystkie 60 checkpointów odtworzyło nowe predykcje dokładnie na GPU; historyczne checkpointy CPU sprawdzone na GPU z maksymalnym błędem około 1.1e−6. Oficjalne mIoU zweryfikowane na wszystkich obrazach, a splity i wybór epoki/progu sprawdzone z zapisanymi historiami. Hashe źródeł i wcześniejszych najlepszych artefaktów pozostały niezmienione.

## Ograniczenia

Mały zbiór (170 obrazów/metodę), istniejące wielokrotnie analizowane foldy. Screening nie jest niezależnym testem. B/C bez awansu nie są ocenione na pozostałych foldach i nie uzasadniają kategorycznego wniosku o wszystkich RGB/SRM architekturach. Użyto trzech filtrów, konkretnej lekkiej CNN i wejścia256; nie pełnego banku SRM ani strojenia rozdzielczości. D ma większą liczbę parametrów niż A; bez kontroli capacity nie przypisujemy każdej różnicy samym residuals. Zachowano najlepsze wcześniejsze rozwiązanie.
