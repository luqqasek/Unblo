# Pretrained image forensics: lokalizacja watermarków

**Wniosek: NO-GO dla dalszej rozbudowy obecnej fuzji.** Uruchomiono wszystkie3 oficjalne modele na510 obrazach (1530 map), a po diagnostykach wykonano E1/E2 DetectiveSAM z63 wewnętrznymi głowicami cross-fit. Baza OOF ma **0,50624 mIoU**, E1 **0,46816**, E2 **0,47946**. E2 wnosi nad własną kontrolę **+0,00352** (CI95 +0,00067…+0,00674), ale względem oryginalnej bazy **−0,02678** (CI95 −0,03559…−0,01855). Największy spadek pochodzi z niestabilnej kalibracji TrustMark, widocznej też w kontroli. Dla StegaStamp E2 ma **0,51629 vs0,51514**, delta+0,00114 z CI95 −0,00368…+0,00630. To słaby komplementarny sygnał w części obrazów, bez potwierdzonego przełomu w lokalizacji. Gate rozpoznaje kierunek w52 informatywnych przypadkach z dokładnością **51,9%**; balanced accuracy3 klas to **34,0%**. E3 nie trenowano.

Eksperyment na 510 oznaczonych obrazach treningowych, 170 na rodzinę. Oficjalna siatka 128×128; trzy istniejące outer-foldy. Nie użyto obrazów testowych ani leaderboardu. Nie generowano nowych danych i nie dostrajano modeli forensic.

## Protokół i źródła

Baza seed42: PixelSeal i TrustMark frozen_extractor/cv3; StegaStamp synthetic_pretrain_2k synthetic_fold{1,2,3}_seed42, z inverse-STN. Baza ta odpowiada recepturze najlepszego zaakceptowanego submission, ale wynik OOF nie jest wynikiem publicznego testu 0,5235. Historyczna średnia trzech seedów StegaStamp 0,50719 również nie jest wynikiem pojedynczego seedu42.

Sprawdzono dokładną zgodność 510 ID, rodzin, foldów i masek z data/train/masks_128.npz. Źródła i SHA256 OOF: [manifest inferencji](results/inference_trufor.json). Oficjalny kod, wagi, licencje i preprocessing: [PROVENANCE.md](PROVENANCE.md). Reprodukcja: [REPRODUCE.md](REPRODUCE.md).

Forensic jest zamrożony. Osobno dla każdej rodziny i outer-foldu progi direct i inverse dobieramy na istniejącym inner_validation (23 obrazy); wybór kierunku selected również korzysta wyłącznie z tego inner_validation. Żaden outer-validation GT nie wybiera progu ani kierunku wdrożeniowego. To porównanie rozwojowe na wielokrotnie oglądanych foldach, nie nowy niezależny test.

Zapisujemy float32 mapy 512×512, area-average do128×128, binarne maski i dopełnienia. Native_binary: oficjalny próg na512, następnie nearest do128. Kalibracja ma 101 kwantyli inner-map, dodatkowe progi0,05…0,95 oraz maski pełną/pustą. TruFor używa RGB/256 i wewnętrznej normalizacji; Mesorch RGB/255 z zewnętrzną normalizacją ImageNet; DetectiveSAM używa oficjalnego prepare_sample bez GT i oficjalnych wewnętrznych strumieni blur/noise. Nie są to nowe przykłady treningowe.

Metryki liczone per obraz, następnie uśredniane. AUC/AP pomijają obrazy z jedną klasą; liczebności zapisane w CSV. Top-K używa prawdziwej powierzchni GT i sprawiedliwie rozdziela remisy. BF1: tolerancja2 piksele siatki128 (8 RGB), zewnętrzna rama obrazu pominięta. Otwory: zamknięte składowe tła z binary_fill_holes; clean-hole recall to odzyskana część pikseli, component recall wymaga odzyskania≥50% składowej. To definicja topologiczna, nie etykieta typu edycji.

Oracle_orientation_fixed wybiera kierunek per obraz z GT przy progu0,5. Oracle_orientation_threshold dodatkowo sprawdza **każdą unikalną wartość** mapy, włącznie z pełną/pustą maską. Oba są tylko diagnostyką. Oracle_empty_full stanowi kontrolę dominującej klasy. Cztery przesunięcia toroidalne zachowują histogram, kształt i powierzchnię mapy, zmieniając położenie; ich oracle AUC porównujemy z nieprzesuniętą mapą.

## Wyniki modeli

### detectivesam

| Wariant | mIoU | FG IoU | Clean IoU | AUC | AP | BF1 |
|---|---|---|---|---|---|---|
| baseline | 0.5062 | 0.4970 | 0.5155 | 0.7000 | 0.6063 | 0.0763 |
| direct_calibrated | 0.2656 | 0.1494 | 0.3817 | 0.4870 | 0.5146 | 0.0341 |
| direct_fixed | 0.2432 | 0.2464 | 0.2400 | 0.4870 | 0.5146 | 0.0960 |
| inverse_calibrated | 0.2896 | 0.3243 | 0.2548 | 0.5130 | 0.5248 | 0.0704 |
| inverse_fixed | 0.3064 | 0.2990 | 0.3137 | 0.5130 | 0.5248 | 0.0944 |
| native_binary | 0.2431 | 0.2462 | 0.2400 | 0.4866 | 0.5036 | 0.0957 |
| native_binary_inverse | 0.3064 | 0.2992 | 0.3137 | 0.5134 | 0.5124 | 0.0931 |
| oracle_empty_full | 0.4547 | 0.4529 | 0.4565 | 0.5000 | 0.4964 | 0.0000 |
| oracle_orientation_fixed | 0.3954 | 0.3943 | 0.3964 | 0.5688 | 0.5464 | 0.0974 |
| oracle_orientation_threshold | 0.5109 | 0.5131 | 0.5088 | 0.6364 | 0.5694 | 0.1153 |
| selected | 0.2909 | 0.2627 | 0.3192 | 0.5140 | 0.5285 | 0.0766 |

Inferencja: mediana 0.142s/obraz; maksimum allocated 470.0MiB. Czasy pełnych przebiegów Mesorch i DetectiveSAM obejmują okres współdzielenia RTX3070; nie są kontrolowanym rankingiem szybkości. Osobne pola reserved, czas per obraz i preprocessing SAM w cache.

Geometria: 59/510 obrazów, na których oracle przy0,5 przekracza oracle empty/full o>0,05. Z tego 35 przekracza także bazę o>0,05; 17 dotyczy ≤5% lub>50% GT. Średni oracle AUC 0.7029, po przesunięciu 0.6620; różnica +0.0409, CI95 [0.02984383611691861, 0.05190022393380026].

Częstość kierunków oracle przy0,5: {'inverse_fixed': 297, 'direct_fixed': 213}. Przy optymalnym progu: {'direct_fixed': 357, 'inverse_fixed': 153}. Remisy przypisujemy direct.

Bramka feasibility: **GO**. Ustalona przed pierwszym pełnym wynikiem: ≥10 komplementarnych obrazów, ≥5 small/large, obecność we wszystkich3 foldach, przewaga oracle AUC nad przesunięciami>0,03. To praktyczne kryterium kontynuacji, nie test hipotezy zerowej ani dowód nieistnienia sygnału.

[Pełne metryki](results/eval_detectivesam/per_image_metrics.csv), [rodziny i powierzchnie](results/eval_detectivesam/summary_by_family_area.csv), [foldy](results/eval_detectivesam/per_fold.csv), [oracle](results/eval_detectivesam/oracle_diagnostics.csv), [kalibracja z ID](results/eval_detectivesam/calibration.json).

![Porównanie detectivesam](results/detectivesam_scatter_examples.png)

### mesorch

| Wariant | mIoU | FG IoU | Clean IoU | AUC | AP | BF1 |
|---|---|---|---|---|---|---|
| baseline | 0.5062 | 0.4970 | 0.5155 | 0.7000 | 0.6063 | 0.0763 |
| direct_calibrated | 0.2755 | 0.2223 | 0.3288 | 0.5470 | 0.5503 | 0.0743 |
| direct_fixed | 0.2553 | 0.0086 | 0.5019 | 0.5470 | 0.5503 | 0.0206 |
| inverse_calibrated | 0.2659 | 0.3276 | 0.2042 | 0.4530 | 0.4973 | 0.0871 |
| inverse_fixed | 0.2539 | 0.4941 | 0.0137 | 0.4530 | 0.4973 | 0.0232 |
| native_binary | 0.2552 | 0.0085 | 0.5019 | 0.4969 | 0.5012 | 0.0204 |
| native_binary_inverse | 0.2539 | 0.4941 | 0.0137 | 0.5031 | 0.4969 | 0.0236 |
| oracle_empty_full | 0.4547 | 0.4529 | 0.4565 | 0.5000 | 0.4964 | 0.0000 |
| oracle_orientation_fixed | 0.4626 | 0.4567 | 0.4685 | 0.5608 | 0.5502 | 0.0223 |
| oracle_orientation_threshold | 0.5155 | 0.5186 | 0.5124 | 0.6477 | 0.5752 | 0.1429 |
| selected | 0.2764 | 0.2775 | 0.2753 | 0.4989 | 0.5290 | 0.0982 |

Inferencja: mediana 0.122s/obraz; maksimum allocated 755.1MiB. Czasy pełnych przebiegów Mesorch i DetectiveSAM obejmują okres współdzielenia RTX3070; nie są kontrolowanym rankingiem szybkości. Osobne pola reserved, czas per obraz i preprocessing SAM w cache.

Geometria: 16/510 obrazów, na których oracle przy0,5 przekracza oracle empty/full o>0,05. Z tego 11 przekracza także bazę o>0,05; 7 dotyczy ≤5% lub>50% GT. Średni oracle AUC 0.6973, po przesunięciu 0.6379; różnica +0.0594, CI95 [0.04756138509620496, 0.07136262925765514].

Częstość kierunków oracle przy0,5: {'direct_fixed': 256, 'inverse_fixed': 254}. Przy optymalnym progu: {'direct_fixed': 392, 'inverse_fixed': 118}. Remisy przypisujemy direct.

Bramka feasibility: **NO-GO**. Ustalona przed pierwszym pełnym wynikiem: ≥10 komplementarnych obrazów, ≥5 small/large, obecność we wszystkich3 foldach, przewaga oracle AUC nad przesunięciami>0,03. To praktyczne kryterium kontynuacji, nie test hipotezy zerowej ani dowód nieistnienia sygnału.

[Pełne metryki](results/eval_mesorch/per_image_metrics.csv), [rodziny i powierzchnie](results/eval_mesorch/summary_by_family_area.csv), [foldy](results/eval_mesorch/per_fold.csv), [oracle](results/eval_mesorch/oracle_diagnostics.csv), [kalibracja z ID](results/eval_mesorch/calibration.json).

![Porównanie mesorch](results/mesorch_scatter_examples.png)

### trufor

| Wariant | mIoU | FG IoU | Clean IoU | AUC | AP | BF1 |
|---|---|---|---|---|---|---|
| baseline | 0.5062 | 0.4970 | 0.5155 | 0.7000 | 0.6063 | 0.0763 |
| direct_calibrated | 0.2807 | 0.2583 | 0.3032 | 0.5706 | 0.5452 | 0.0680 |
| direct_fixed | 0.2677 | 0.0530 | 0.4823 | 0.5706 | 0.5452 | 0.0507 |
| inverse_calibrated | 0.2574 | 0.2212 | 0.2936 | 0.4294 | 0.4916 | 0.0425 |
| inverse_fixed | 0.2512 | 0.4663 | 0.0361 | 0.4294 | 0.4916 | 0.0545 |
| native_binary | 0.2676 | 0.0530 | 0.4823 | 0.5101 | 0.5103 | 0.0510 |
| native_binary_inverse | 0.2512 | 0.4663 | 0.0361 | 0.4899 | 0.4959 | 0.0548 |
| oracle_empty_full | 0.4547 | 0.4529 | 0.4565 | 0.5000 | 0.4964 | 0.0000 |
| oracle_orientation_fixed | 0.4458 | 0.4459 | 0.4457 | 0.5463 | 0.5361 | 0.0522 |
| oracle_orientation_threshold | 0.5050 | 0.5105 | 0.4996 | 0.6327 | 0.5633 | 0.1083 |
| selected | 0.2745 | 0.2766 | 0.2724 | 0.5007 | 0.5224 | 0.0693 |

Inferencja: mediana 0.213s/obraz; maksimum allocated 986.2MiB. Czasy pełnych przebiegów Mesorch i DetectiveSAM obejmują okres współdzielenia RTX3070; nie są kontrolowanym rankingiem szybkości. Osobne pola reserved, czas per obraz i preprocessing SAM w cache.

Geometria: 33/510 obrazów, na których oracle przy0,5 przekracza oracle empty/full o>0,05. Z tego 20 przekracza także bazę o>0,05; 6 dotyczy ≤5% lub>50% GT. Średni oracle AUC 0.6890, po przesunięciu 0.6608; różnica +0.0281, CI95 [0.01643345256663323, 0.03964923641363645].

Częstość kierunków oracle przy0,5: {'direct_fixed': 258, 'inverse_fixed': 252}. Przy optymalnym progu: {'direct_fixed': 420, 'inverse_fixed': 90}. Remisy przypisujemy direct.

Bramka feasibility: **NO-GO**. Ustalona przed pierwszym pełnym wynikiem: ≥10 komplementarnych obrazów, ≥5 small/large, obecność we wszystkich3 foldach, przewaga oracle AUC nad przesunięciami>0,03. To praktyczne kryterium kontynuacji, nie test hipotezy zerowej ani dowód nieistnienia sygnału.

[Pełne metryki](results/eval_trufor/per_image_metrics.csv), [rodziny i powierzchnie](results/eval_trufor/summary_by_family_area.csv), [foldy](results/eval_trufor/per_fold.csv), [oracle](results/eval_trufor/oracle_diagnostics.csv), [kalibracja z ID](results/eval_trufor/calibration.json).

![Porównanie trufor](results/trufor_scatter_examples.png)

## Rodziny i najtrudniejsze obszary

| Model | Rodzina | Wariant | mIoU | AUC | AP | BF1 |
|---|---|---|---|---|---|---|
| detectivesam | pixelseal | baseline | 0.5078 | 0.7087 | 0.6306 | 0.0765 |
| detectivesam | pixelseal | selected | 0.2719 | 0.4883 | 0.5231 | 0.0798 |
| detectivesam | trustmark | baseline | 0.4958 | 0.6435 | 0.5861 | 0.0632 |
| detectivesam | trustmark | selected | 0.2832 | 0.4833 | 0.5254 | 0.0637 |
| detectivesam | stegastamp | baseline | 0.5151 | 0.7479 | 0.6021 | 0.0893 |
| detectivesam | stegastamp | selected | 0.3176 | 0.5704 | 0.5369 | 0.0864 |
| mesorch | pixelseal | baseline | 0.5078 | 0.7087 | 0.6306 | 0.0765 |
| mesorch | pixelseal | selected | 0.2706 | 0.4906 | 0.5082 | 0.0988 |
| mesorch | trustmark | baseline | 0.4958 | 0.6435 | 0.5861 | 0.0632 |
| mesorch | trustmark | selected | 0.2657 | 0.4852 | 0.5368 | 0.1019 |
| mesorch | stegastamp | baseline | 0.5151 | 0.7479 | 0.6021 | 0.0893 |
| mesorch | stegastamp | selected | 0.2928 | 0.5210 | 0.5421 | 0.0940 |
| trufor | pixelseal | baseline | 0.5078 | 0.7087 | 0.6306 | 0.0765 |
| trufor | pixelseal | selected | 0.2933 | 0.5506 | 0.5378 | 0.0744 |
| trufor | trustmark | baseline | 0.4958 | 0.6435 | 0.5861 | 0.0632 |
| trufor | trustmark | selected | 0.2567 | 0.4510 | 0.5058 | 0.0765 |
| trufor | stegastamp | baseline | 0.5151 | 0.7479 | 0.6021 | 0.0893 |
| trufor | stegastamp | selected | 0.2734 | 0.5004 | 0.5236 | 0.0571 |

### Małe watermarki ≤5%

| Model | Wariant | N | FG IoU | AP | Top-K oracle | Miss rate | FP area |
|---|---|---|---|---|---|---|---|
| detectivesam | baseline | 133 | 0.0221 | 0.0653 | 0.0514 | 0.8421 | 0.0277 |
| detectivesam | direct_fixed | 133 | 0.0189 | 0.0290 | 0.0144 | 0.1429 | 0.5810 |
| detectivesam | inverse_fixed | 133 | 0.0219 | 0.0332 | 0.0163 | 0.2932 | 0.3990 |
| detectivesam | selected | 133 | 0.0147 | 0.0269 | 0.0111 | 0.3759 | 0.3610 |
| mesorch | baseline | 133 | 0.0221 | 0.0653 | 0.0514 | 0.8421 | 0.0277 |
| mesorch | direct_fixed | 133 | 0.0000 | 0.0709 | 0.0544 | 0.9925 | 0.0048 |
| mesorch | inverse_fixed | 133 | 0.0201 | 0.0248 | 0.0161 | 0.0000 | 0.9752 |
| mesorch | selected | 133 | 0.0256 | 0.0660 | 0.0534 | 0.1203 | 0.4521 |
| trufor | baseline | 133 | 0.0221 | 0.0653 | 0.0514 | 0.8421 | 0.0277 |
| trufor | direct_fixed | 133 | 0.0112 | 0.0521 | 0.0340 | 0.8271 | 0.0521 |
| trufor | inverse_fixed | 133 | 0.0203 | 0.0247 | 0.0099 | 0.0075 | 0.9279 |
| trufor | selected | 133 | 0.0198 | 0.0397 | 0.0245 | 0.2556 | 0.4394 |

### Duże watermarki >50%

| Model | Wariant | N | Clean IoU | Internal BF1 | Hole recall | Clean FPR |
|---|---|---|---|---|---|---|
| detectivesam | baseline | 255 | 0.1380 | 0.0415 | 0.1183 | 0.7769 |
| detectivesam | direct_fixed | 255 | 0.0695 | 0.0300 | 0.4363 | 0.4680 |
| detectivesam | inverse_fixed | 255 | 0.0930 | 0.0444 | 0.5637 | 0.5320 |
| detectivesam | selected | 255 | 0.0834 | 0.0261 | 0.5732 | 0.4722 |
| mesorch | baseline | 255 | 0.1380 | 0.0415 | 0.1183 | 0.7769 |
| mesorch | direct_fixed | 255 | 0.0929 | 0.0005 | 0.9895 | 0.0264 |
| mesorch | inverse_fixed | 255 | 0.0240 | 0.0122 | 0.0105 | 0.9736 |
| mesorch | selected | 255 | 0.0670 | 0.0417 | 0.4503 | 0.5421 |
| trufor | baseline | 255 | 0.1380 | 0.0415 | 0.1183 | 0.7769 |
| trufor | direct_fixed | 255 | 0.0956 | 0.0045 | 0.9195 | 0.0475 |
| trufor | inverse_fixed | 255 | 0.0170 | 0.0259 | 0.0805 | 0.9525 |
| trufor | selected | 255 | 0.0633 | 0.0220 | 0.4676 | 0.5511 |

## Komplementarność, foldy i niepewność

Failure: baza mIoU<0,4; poprawa oznacza delta>0,05. Previously-good: baza≥0,5; uszkodzenie oznacza delta<−0,05. To opisowe progi diagnostyczne. Korelacja błędów jest Pearsonem masek błędu per obraz; brak wariancji daje NaN. Odmienność błędów sama nie oznacza przydatności.

| Model | Korelacja błędów | IoU błędów | Poprawione failures | Uszkodzone dobre | Delta mIoU |
|---|---|---|---|---|---|
| detectivesam | -0.0064 | 0.1662 | 3 | 131 | -0.2153 |
| mesorch | 0.0076 | 0.1483 | 2 | 137 | -0.2299 |
| trufor | 0.0279 | 0.1679 | 7 | 142 | -0.2318 |

detectivesam: selected−baza -0.21532, parowany bootstrap obrazów CI95 [-0.23149772643426933, -0.1984120623230572]. 5000 resampli, seed20261121; przedział warunkowy względem zapisanych modeli, nie obejmuje losowości treningu ani selekcji kierunku badawczego.

mesorch: selected−baza -0.22985, parowany bootstrap obrazów CI95 [-0.2460196986363931, -0.21391140521106825]. 5000 resampli, seed20261121; przedział warunkowy względem zapisanych modeli, nie obejmuje losowości treningu ani selekcji kierunku badawczego.

trufor: selected−baza -0.23178, parowany bootstrap obrazów CI95 [-0.24823850838626896, -0.21517869781537693]. 5000 resampli, seed20261121; przedział warunkowy względem zapisanych modeli, nie obejmuje losowości treningu ani selekcji kierunku badawczego.

| Model | Rodzina | Fold | Wariant | mIoU | AP |
|---|---|---|---|---|---|
| detectivesam | pixelseal | 1 | baseline | 0.5191 | 0.6021 |
| detectivesam | pixelseal | 1 | selected | 0.2720 | 0.4837 |
| detectivesam | pixelseal | 2 | baseline | 0.4773 | 0.6147 |
| detectivesam | pixelseal | 2 | selected | 0.2798 | 0.5314 |
| detectivesam | pixelseal | 3 | baseline | 0.5272 | 0.6757 |
| detectivesam | pixelseal | 3 | selected | 0.2638 | 0.5547 |
| detectivesam | stegastamp | 1 | baseline | 0.5182 | 0.6787 |
| detectivesam | stegastamp | 1 | selected | 0.3773 | 0.6151 |
| detectivesam | stegastamp | 2 | baseline | 0.4979 | 0.5918 |
| detectivesam | stegastamp | 2 | selected | 0.2477 | 0.5037 |
| detectivesam | stegastamp | 3 | baseline | 0.5297 | 0.5348 |
| detectivesam | stegastamp | 3 | selected | 0.3280 | 0.4912 |
| detectivesam | trustmark | 1 | baseline | 0.5129 | 0.5284 |
| detectivesam | trustmark | 1 | selected | 0.2797 | 0.4806 |
| detectivesam | trustmark | 2 | baseline | 0.4810 | 0.6911 |
| detectivesam | trustmark | 2 | selected | 0.3006 | 0.6341 |
| detectivesam | trustmark | 3 | baseline | 0.4935 | 0.5380 |
| detectivesam | trustmark | 3 | selected | 0.2691 | 0.4602 |
| mesorch | pixelseal | 1 | baseline | 0.5191 | 0.6021 |
| mesorch | pixelseal | 1 | selected | 0.2739 | 0.5050 |
| mesorch | pixelseal | 2 | baseline | 0.4773 | 0.6147 |
| mesorch | pixelseal | 2 | selected | 0.2575 | 0.4833 |
| mesorch | pixelseal | 3 | baseline | 0.5272 | 0.6757 |
| mesorch | pixelseal | 3 | selected | 0.2806 | 0.5369 |
| mesorch | stegastamp | 1 | baseline | 0.5182 | 0.6787 |
| mesorch | stegastamp | 1 | selected | 0.2915 | 0.6449 |
| mesorch | stegastamp | 2 | baseline | 0.4979 | 0.5918 |
| mesorch | stegastamp | 2 | selected | 0.2501 | 0.4646 |
| mesorch | stegastamp | 3 | baseline | 0.5297 | 0.5348 |
| mesorch | stegastamp | 3 | selected | 0.3376 | 0.5163 |
| mesorch | trustmark | 1 | baseline | 0.5129 | 0.5284 |
| mesorch | trustmark | 1 | selected | 0.3033 | 0.5293 |
| mesorch | trustmark | 2 | baseline | 0.4810 | 0.6911 |
| mesorch | trustmark | 2 | selected | 0.2362 | 0.5998 |
| mesorch | trustmark | 3 | baseline | 0.4935 | 0.5380 |
| mesorch | trustmark | 3 | selected | 0.2575 | 0.4802 |
| trufor | pixelseal | 1 | baseline | 0.5191 | 0.6021 |
| trufor | pixelseal | 1 | selected | 0.2922 | 0.5196 |
| trufor | pixelseal | 2 | baseline | 0.4773 | 0.6147 |
| trufor | pixelseal | 2 | selected | 0.2902 | 0.5474 |
| trufor | pixelseal | 3 | baseline | 0.5272 | 0.6757 |
| trufor | pixelseal | 3 | selected | 0.2976 | 0.5466 |
| trufor | stegastamp | 1 | baseline | 0.5182 | 0.6787 |
| trufor | stegastamp | 1 | selected | 0.2773 | 0.5355 |
| trufor | stegastamp | 2 | baseline | 0.4979 | 0.5918 |
| trufor | stegastamp | 2 | selected | 0.2583 | 0.5284 |
| trufor | stegastamp | 3 | baseline | 0.5297 | 0.5348 |
| trufor | stegastamp | 3 | selected | 0.2848 | 0.5067 |
| trufor | trustmark | 1 | baseline | 0.5129 | 0.5284 |
| trufor | trustmark | 1 | selected | 0.2800 | 0.4747 |
| trufor | trustmark | 2 | baseline | 0.4810 | 0.6911 |
| trufor | trustmark | 2 | selected | 0.2402 | 0.6083 |
| trufor | trustmark | 3 | baseline | 0.4935 | 0.5380 |
| trufor | trustmark | 3 | selected | 0.2497 | 0.4332 |

## Decyzja o fuzji

Bramkę przechodzi DetectiveSAM. Wykonano E1/E2 i dopasowane kontrole na ścisłych wewnętrznych cross-fitted mapach; wyniki i końcowa decyzja w sekcji Faza E poniżej.

## Odpowiedzi na osiem pytań

1. **Czy pretrained manipulation models znajdują użyteczne regiony?** Zdarzają się pojedyncze trafienia geometryczne; ich liczbę ponad empty/full i bazę podają bramki powyżej. To nie dowód na użyteczność całego systemu.
2. **Czy granice odpowiadają granicom watermarku?** Oceniamy BF1 i wewnętrzny BF1 w tabelach; galerie pokazują zarówno trafienia, jak i niepowiązane obiekty. Wysoki oracle mIoU bez przewagi nad pustą/pełną maską nie jest takim dowodem.
3. **Direct czy inverse?** Nie ma bezpiecznej reguły „manipulacja=watermark”. Histogram zwycięstw oracle pokazuje rozbieżność, lecz jest zależny od GT i powierzchni. Metadata nie zawiera wiarygodnych etykiet wszystkich czterech typów edycji, więc nie przypisujemy im wyników na podstawie domysłów.
4. **Czy kierunek można wybrać bez GT?** Przetestowano rodzinny wybór kierunku i progu na inner_validation. Wynik selected jest wdrożeniowy w sensie braku dostępu do outer GT, ale nie jest image-level gate. W fazie E przetestowano również miękki gate per obraz; jego wynik należy oceniać względem watermark-only control, nie oracle.
5. **Czy połączenie poprawia prawdziwą lokalizację?** Oracle najlepszego systemu jest wyłącznie górnym diagnostycznym wskazaniem. Dla DetectiveSAM wykonano E1/E2; wyniki poniżej pokazują, ile wnosi forensic ponad samą kalibrację. Nie testowano wszystkich możliwych fuzji.
6. **Który model jest najlepszy na rodzinie?** Ranking poniżej dotyczy selected i jest opisem po obejrzeniu OOF, nie nową regułą wyboru modelu do inferencji. Ranking AP może się różnić od mIoU.
7. **Czy wzrost jest wiarygodny między foldami?** Foldy i parowane CI podano powyżej; brak nowej niezależnej walidacji oraz tylko seed42 dla porównania baz ograniczają uogólnienia.
8. **Czy rozwijać kierunek?** Próbę E1/E2 wykonano. Rekomendacja: zakończyć tę ścieżkę jako priorytet; nie potwierdzono wymaganej przewagi nad dopasowanymi kontrolami uzasadniającej E3.

| Rodzina | Najlepszy selected model | mIoU | AUC | AP |
|---|---|---|---|---|
| pixelseal | trufor | 0.2933 | 0.5506 | 0.5378 |
| stegastamp | detectivesam | 0.3176 | 0.5704 | 0.5369 |
| trustmark | detectivesam | 0.2832 | 0.4833 | 0.5254 |

## Materiały i ograniczenia

[Galeria przypadków](results/gallery/index.html), [manifest wyboru przykładów](results/gallery/cases.json), [komplementarność](results/complementarity_summary.csv), [pełne przedziały powierzchni](results/summary_by_family_area.csv). Kategorie obejmują małe regiony, otwory, poprawy, regresje, wysoką pewność z błędem i niejednoznaczną orientację. Bez GT manipulations nie można rozstrzygnąć, czy model błędnie wykrył edycję, czy poprawnie wykrył edycję niezwiązaną z watermarkiem.

TruFor dostarcza confidence i global detection score. Mesorch i użyty interfejs DetectiveSAM nie udostępniają osobnej skalibrowanej mapy pewności; nie zastępujemy jej arbitralną funkcją prawdopodobieństwa. Panele „confidence_wrong_or_unavailable” w tych modelach nie oznaczają zweryfikowanej wysokiej pewności.

Modele mogły widzieć źródłowe fotografie w swoim pretrainingu; nie mamy identyfikatorów źródeł konkursowych umożliwiających audyt tego nakładania. Nie zmieniono dotychczasowych checkpointów ani wyników. Kod oficjalny i checkpointy przechowywane w ignorowanym results; same źródła eksperymentu są wersjonowalne.



## Faza E: uczona fuzja Detective SAM

Fazę E uruchomiono po zakończeniu C/D: jedynie Detective SAM przeszedł bramkę. Powstały63 małe głowice baseline do generowania poprawnych wewnętrznych map. Ekstraktory watermarku i wszystkie modele forensic pozostały zamrożone. StegaStamp zachował inverse-STN i inicjalizację istniejącym checkpointem pretrain_seed42; nie wytwarzano ani nie czytano nowych syntetyków.

Każdy outer-fold ma osobny meta-train i meta-validation zgodny z istniejącym inner_train/inner_validation. Mapy meta-train są3-way cross-fitted **wyłącznie w meta-train**; model generujący mapy meta-validation też nigdy go nie widział. Po wyborze parametrów fuzja jest dopasowana ponownie na oddzielnych3-way cross-fitted mapach całego outer-train. Każda głowica wybiera epokę na swoim własnym80/20 podziale, potem uczy się od nowa. Żadna mapa fitting/calibration nie ma producenta uczonego lub wybieranego na outer-validation. [Audyt63 modeli](results/verification.json).

Mapy wewnętrzne pochodzą z mniejszych zbiorów niż docelowa głowica. Wewnętrzna selekcja głowicy używa prostego80/20 także dla StegaStamp, podczas gdy zapisany zewnętrzny baseline używał stabilnej selekcji3-fold. To ograniczenie dopasowania rozkładu meta-danych; kontrola i forensic dostają dokładnie te same mapy. Zewnętrzne predykcje baseline są oryginalnymi zachowanymi OOF.

**E1:** sigmoid(b+w1·logit(W)+w2·logit(F)), clipping1e−5. Kontrola ma tyle samo3 parametrów: sigmoid(b+w1·logit(W)+w2·logit(W)²/5). Strata BCE na1024 deterministycznie próbkowanych pikselach na obraz; równa waga obrazów; kara L2 {0,001;0,01;0,1} względem tożsamości. Dobór kary i progu0,05…0,95 wyłącznie na meta-validation.

**E2:** miękki multinomial logistic gate z48 nominalnymi parametrami: po7 statystyk W/F i globalne bit accuracy znanej wiadomości. Statystyki: mean/std/kwantyle10,50,90/area@0,5/entropy. Klasy: zachowaj W, użyj F, użyj1−F. Etykiety treningowe wyznacza przewaga odpowiedniej akcji nad W>0,02 mIoU. Finalna mapa jest miękką mieszanką z osłabieniem α∈{0,25;0,5;1}. C∈{0,01;0,1;1} i próg wybiera meta-validation. Kontrola o tym samym rozmiarze używa wyłącznie statystyk W, ich kwadratów i bit accuracy; miesza W/pustą/pełną maskę. To kontrola zdolności kalibracji i przewidywania dominującej klasy.

Wyniki E2 nie wykorzystują per-image GT do wyboru kierunku przy inferencji. GT jest wyłącznie celem uczenia gate na dopuszczonym treningu. Nie używamy full_watermark jako wejścia, bo ta informacja jest treningowym metadata.

| Wariant | mIoU | FG IoU | Clean IoU | AUC | AP | BF1 |
|---|---|---|---|---|---|---|
| E1_control | 0.4708 | 0.4256 | 0.5160 | 0.7007 | 0.6064 | 0.0714 |
| E1_forensic | 0.4682 | 0.4245 | 0.5118 | 0.6982 | 0.6057 | 0.0712 |
| E2_control | 0.4759 | 0.4338 | 0.5181 | 0.7000 | 0.6063 | 0.0707 |
| E2_forensic | 0.4795 | 0.4367 | 0.5222 | 0.6934 | 0.6051 | 0.0755 |
| baseline | 0.5062 | 0.4970 | 0.5155 | 0.7000 | 0.6063 | 0.0763 |

### Porównania parowane

| Rodzina | Wariant | Kontrola | Delta mIoU | CI95 | Dodatnie obrazy | Ujemne obrazy |
|---|---|---|---|---|---|---|
| all | E1_forensic | baseline | -0.0381 | [-0.04900561177282351, -0.02813573458393573] | 194 | 254 |
| all | E1_forensic | E1_control | -0.0026 | [-0.005136105084529719, -0.00016533917742685706] | 194 | 190 |
| all | E2_forensic | baseline | -0.0268 | [-0.03558759906370073, -0.018547499965282134] | 188 | 238 |
| all | E2_forensic | E2_control | 0.0035 | [0.0006650559154750846, 0.0067448087137510675] | 236 | 170 |
| pixelseal | E1_forensic | baseline | 0.0022 | [-0.004485357171741735, 0.009163416543876273] | 85 | 75 |
| pixelseal | E1_forensic | E1_control | -0.0012 | [-0.0028631032104310303, 0.0005435778760891613] | 68 | 87 |
| pixelseal | E2_forensic | baseline | 0.0009 | [-0.006953735414884074, 0.009289992165956216] | 85 | 75 |
| pixelseal | E2_forensic | E2_control | -0.0009 | [-0.007088664502867076, 0.004680946978515862] | 68 | 85 |
| trustmark | E1_forensic | baseline | -0.1117 | [-0.1394991762569029, -0.08457425754626421] | 34 | 99 |
| trustmark | E1_forensic | E1_control | -0.0072 | [-0.013059449308517782, -0.0007734183002321463] | 37 | 45 |
| trustmark | E2_forensic | baseline | -0.0824 | [-0.10609793717216538, -0.06038091921735687] | 31 | 88 |
| trustmark | E2_forensic | E2_control | 0.0037 | [0.002582351461276377, 0.004931601262205537] | 79 | 30 |
| stegastamp | E1_forensic | baseline | -0.0047 | [-0.012474840856931815, 0.0028187581597113125] | 75 | 80 |
| stegastamp | E1_forensic | E1_control | 0.0005 | [-0.0039008101990276557, 0.004459924784337677] | 89 | 58 |
| stegastamp | E2_forensic | baseline | 0.0011 | [-0.003676792885163461, 0.006295360628467847] | 72 | 75 |
| stegastamp | E2_forensic | E2_control | 0.0078 | [0.0021347444081401506, 0.016113530141000094] | 89 | 55 |

### Czy gate rozpoznaje interpretację bez GT?

Klasy0/1/2: zachowaj baseline / direct / inverse; target przyznaje direct/inverse tylko przy przewadze>0,02 nad baseline przy progu0,5. Majority control wybiera najczęstszą klasę z odpowiedniego outer-train. Accuracy może być wysokie przez częste „zachowaj baseline”; balanced accuracy i dokładność kierunku wśród informatywnych przypadków są osobnymi diagnostykami. Wszystkie targety zewnętrzne użyto tylko do tej oceny.

| Rodzina | Accuracy | Balanced accuracy | Majority accuracy | Informative N | Kierunek w informative |
|---|---|---|---|---|---|
| all | 0.8922 | 0.3399 | 0.8980 | 52 | 0.5192 |
| pixelseal | 0.8647 | 0.3289 | 0.8765 | 21 | 0.4286 |
| trustmark | 0.9176 | 0.3333 | 0.9176 | 14 | 0.2857 |
| stegastamp | 0.8941 | 0.3512 | 0.9000 | 17 | 0.8235 |

[Macierze pomyłek i log-loss](results/fusion/orientation_summary.json), [per-image gate](results/fusion/orientation_per_image.csv), [rozkłady map wewnętrznych i zewnętrznych](results/fusion/map_distribution_shift.csv).

### Małe ≤5%

| Wariant | FG IoU | AP | Top-K | Complete miss | FP area |
|---|---|---|---|---|---|
| E1_control | 0.0204 | 0.0656 | 0.0514 | 0.8421 | 0.0261 |
| E1_forensic | 0.0225 | 0.0643 | 0.0499 | 0.8271 | 0.0382 |
| E2_control | 0.0208 | 0.0653 | 0.0514 | 0.8647 | 0.0179 |
| E2_forensic | 0.0217 | 0.0664 | 0.0518 | 0.8722 | 0.0188 |
| baseline | 0.0221 | 0.0653 | 0.0514 | 0.8421 | 0.0277 |

### Duże >50%

| Wariant | Clean IoU | Internal BF1 | Hole recall | Clean FPR |
|---|---|---|---|---|
| E1_control | 0.1365 | 0.0351 | 0.2730 | 0.6356 |
| E1_forensic | 0.1385 | 0.0367 | 0.2836 | 0.6285 |
| E2_control | 0.1356 | 0.0382 | 0.2491 | 0.6553 |
| E2_forensic | 0.1416 | 0.0412 | 0.2499 | 0.6426 |
| baseline | 0.1380 | 0.0415 | 0.1183 | 0.7769 |

### Wyniki per fold

| Rodzina | Fold | Wariant | mIoU | AP |
|---|---|---|---|---|
| pixelseal | 1 | E1_control | 0.5211 | 0.6021 |
| pixelseal | 1 | E1_forensic | 0.5213 | 0.6025 |
| pixelseal | 1 | E2_control | 0.5214 | 0.6021 |
| pixelseal | 1 | E2_forensic | 0.5147 | 0.5944 |
| pixelseal | 1 | baseline | 0.5191 | 0.6021 |
| pixelseal | 2 | E1_control | 0.4925 | 0.6147 |
| pixelseal | 2 | E1_forensic | 0.4895 | 0.6157 |
| pixelseal | 2 | E2_control | 0.4843 | 0.6147 |
| pixelseal | 2 | E2_forensic | 0.4908 | 0.6100 |
| pixelseal | 2 | baseline | 0.4773 | 0.6147 |
| pixelseal | 3 | E1_control | 0.5200 | 0.6757 |
| pixelseal | 3 | E1_forensic | 0.5190 | 0.6754 |
| pixelseal | 3 | E2_control | 0.5233 | 0.6757 |
| pixelseal | 3 | E2_forensic | 0.5208 | 0.6748 |
| pixelseal | 3 | baseline | 0.5272 | 0.6757 |
| stegastamp | 1 | E1_control | 0.5035 | 0.6787 |
| stegastamp | 1 | E1_forensic | 0.4971 | 0.6804 |
| stegastamp | 1 | E2_control | 0.4954 | 0.6787 |
| stegastamp | 1 | E2_forensic | 0.5103 | 0.6788 |
| stegastamp | 1 | baseline | 0.5182 | 0.6787 |
| stegastamp | 2 | E1_control | 0.4932 | 0.5918 |
| stegastamp | 2 | E1_forensic | 0.4957 | 0.5911 |
| stegastamp | 2 | E2_control | 0.4977 | 0.5918 |
| stegastamp | 2 | E2_forensic | 0.5025 | 0.5918 |
| stegastamp | 2 | baseline | 0.4979 | 0.5918 |
| stegastamp | 3 | E1_control | 0.5334 | 0.5348 |
| stegastamp | 3 | E1_forensic | 0.5389 | 0.5409 |
| stegastamp | 3 | E2_control | 0.5328 | 0.5348 |
| stegastamp | 3 | E2_forensic | 0.5364 | 0.5372 |
| stegastamp | 3 | baseline | 0.5297 | 0.5348 |
| trustmark | 1 | E1_control | 0.5042 | 0.5290 |
| trustmark | 1 | E1_forensic | 0.5093 | 0.5279 |
| trustmark | 1 | E2_control | 0.5078 | 0.5284 |
| trustmark | 1 | E2_forensic | 0.5084 | 0.5280 |
| trustmark | 1 | baseline | 0.5129 | 0.5284 |
| trustmark | 2 | E1_control | 0.1972 | 0.6911 |
| trustmark | 2 | E1_forensic | 0.1972 | 0.6799 |
| trustmark | 2 | E2_control | 0.2406 | 0.6911 |
| trustmark | 2 | E2_forensic | 0.2490 | 0.6911 |
| trustmark | 2 | baseline | 0.4810 | 0.6911 |
| trustmark | 3 | E1_control | 0.4740 | 0.5382 |
| trustmark | 3 | E1_forensic | 0.4471 | 0.5367 |
| trustmark | 3 | E2_control | 0.4820 | 0.5380 |
| trustmark | 3 | E2_forensic | 0.4841 | 0.5386 |
| trustmark | 3 | baseline | 0.4935 | 0.5380 |

### Decyzja końcowa

**NO-GO dla dalszej rozbudowy forensic fusion.** Proste warianty nie spełniły bramki awansu do E3; CNN nie trenowano. Bramka E3: delta mIoU nad dopasowaną kontrolą>0,005, dolna granica parowanego CI>0, co najmniej2 dodatnie foldy oraz small AP+0,005 lub large clean IoU+0,01. Przed pełnym wynikiem, po zaobserwowaniu niestabilności TrustMark, dopisano wymóg dodatniej delty względem oryginalnego baseline. Sama poprawa wobec słabej kontroli nie uzasadnia E3. Ten dodatkowy wymóg nie jest używany do wyboru wariantu ani progów.

Wybór DetectiveSAM po wcześniejszych diagnostykach i wielokrotne używanie tych samych outer-foldów oznaczają selekcję rozwojową. Bootstrap resampluje obrazy, nie treningi; przedstawiono seed42, nie niezależne potwierdzenie na3 seedach ani nowym zbiorze. Wynik nie uzasadnia wniosków o wszystkich możliwych fuzjach czy wszystkich modelach forensics.

[Per-image fuzji](results/fusion/per_image_metrics.csv), [dobór ustawień](results/fusion/configurations.json), [sparowane delty](results/fusion/paired_deltas.csv), [galeria fuzji](results/fusion/gallery/index.html).
