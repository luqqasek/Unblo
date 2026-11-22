# Etap 0: małe regiony aktualnego message_localizer

Pomiar 2026-11-22, lokalna RTX 3070 8 GB. 170 obrazów PixelSeal i 170 TrustMark, ten sam seed42 i trzy foldy co baza #2163. **Bez nowego treningu i submission.** Najlepszy zaakceptowany wynik publiczny pozostaje 0,6145.

## Wniosek

**Samo przesunięcie progu lub hysteresis nie naprawia małych znaków.** Globalny oracle mIoU poprawia się najwyżej o około 0,0002 PS / 0,0006 TM w sprawdzonej rodzinie postprocessingu. Dla PS≤1% nawet próg dobierany osobno dla każdego obrazu daje średnie FG IoU tylko 0,0089. W TM jest więcej sygnału rankingowego, ale za mało do dużego zysku samą kalibracją.

**Dopasowanie cropu ma znaczenie:** TM 1–5% z niemal prostokątną maską daje 99,05% bit accuracy po wycięciu bbox, a nieregularne maski 48,94%. Stałe okna 128/192 pozostają blisko przypadku. Uzasadnia to osobne reżimy lokalnego osadzania i zachowanych fragmentów oraz kontrolowany test dopasowanego zoom-in. Nie jest to dowód odtworzenia historii edycji każdego obrazu.

**Sygnał PS jest słaby i zależny od JND:** w kontrolach embed 64×64 odsetek pikseli z dokładnym ΔRGB=0 to 51,45%; w dolnym kwartylu JND 74,88%. Chroma ma niewielką energię względem Y. To uzasadnienie testu pomocniczego residual head w Y, nie dowód jego przyszłej poprawy mIoU.

## 1. Wyniki bazowe i kontrola protokołu

Sześć map FP32 sprawdzono po nazwach, foldach i oficjalnym GT128; każdy obraz występuje dokładnie raz, outer-train i outer-validation są rozłączne. mIoU, FG/clean IoU, AUC, AP i puste predykcje odtworzone z map zgadzają się z zapisanymi metrykami. Hashe plików są w `results/oof.json`.

Koszyki: (0,1%], (1%,5%], (5%,20%], >20%, według oficjalnego GT128. Clean holes to `full_watermark=true` i niejednorodna maska; jest dodatkową, nakładającą się kohortą. Dla niej raportujemy też IoU klasy clean, bo samo wysokie FG IoU ukrywa niewykrywanie dziur.

| Metoda | Koszyk | n | mIoU | FG IoU | Clean IoU |
|---|---|---|---|---|---|
| pixelseal | wszystkie | 170 | 0.57192 | 0.55386 | 0.58997 |
| pixelseal | ≤1% | 17 | 0.49435 | 0.00000 | 0.98870 |
| pixelseal | 1–5% | 26 | 0.49937 | 0.02435 | 0.97439 |
| pixelseal | 5–20% | 30 | 0.60026 | 0.29500 | 0.90553 |
| pixelseal | >20% | 97 | 0.59619 | 0.87292 | 0.31946 |
| pixelseal | clean holes | 85 | 0.58003 | 0.90887 | 0.25119 |
| trustmark | wszystkie | 170 | 0.57872 | 0.58062 | 0.57682 |
| trustmark | ≤1% | 9 | 0.49705 | 0.00000 | 0.99409 |
| trustmark | 1–5% | 36 | 0.51378 | 0.06082 | 0.96673 |
| trustmark | 5–20% | 25 | 0.64334 | 0.35557 | 0.93112 |
| trustmark | >20% | 100 | 0.59330 | 0.87627 | 0.31032 |
| trustmark | clean holes | 85 | 0.56533 | 0.90917 | 0.22149 |

## 2. Czy model widzi sygnał?

Wszystkie średnie są macro po obrazach. PR-AUC(AP) oznacza nieinterpolowane average precision; dodatkową całkę trapezową zapisano w JSON/CSV. Max GT i max tła to średnie z maksimum w danym obszarze każdego obrazu, nie średnia wszystkich pikseli.

| Metoda | Koszyk | n | ROC-AUC | PR-AUC(AP) | Puste t=.5 | Max GT | Max poza GT | Udział FG / losowy AP |
|---|---|---|---|---|---|---|---|---|
| pixelseal | ≤1% | 17 | 0.4792 | 0.0058 | 88.2% | 0.0471 | 0.3290 | 0.0055 |
| pixelseal | 1–5% | 26 | 0.5503 | 0.0990 | 69.2% | 0.1468 | 0.3264 | 0.0243 |
| trustmark | ≤1% | 9 | 0.6475 | 0.0373 | 88.9% | 0.0988 | 0.3480 | 0.0059 |
| trustmark | 1–5% | 36 | 0.6696 | 0.1591 | 50.0% | 0.2546 | 0.5551 | 0.0284 |

### Oracle dla każdego koszyka

Próg wspólny przeszukano dokładnie po wszystkich różnych wartościach prawdopodobieństwa, maksymalizując średnie FG IoU. Hysteresis: 8-sąsiedztwo, 85 wcześniej ustalonych par low/high; maksimum na tej siatce nie jest ścisłym supremum wszystkich par. Osobno pokazano silniejszy oracle progu per obraz. **Każdy oracle korzysta z GT ocenianych obrazów; nie jest wynikiem walidacyjnym nowej metody.** Optimum FG może drastycznie pogarszać clean IoU i oficjalne mIoU.

| Metoda | Koszyk | FG t=.5 | Próg oracle | FG oracle | mIoU oracle | Hyst low/high | FG hyst | mIoU hyst | FG oracle per obraz |
|---|---|---|---|---|---|---|---|---|---|
| pixelseal | ≤1% | 0.0000 | 0.0148 | 0.0058 | 0.1590 | 0.001/0.05 | 0.0055 | 0.0027 | 0.0089 |
| pixelseal | 1–5% | 0.0243 | 0.1030 | 0.0463 | 0.4916 | 0.1/0.2 | 0.0469 | 0.4935 | 0.0879 |
| pixelseal | 5–20% | 0.2950 | 0.3054 | 0.3058 | 0.6017 | 0.3/0.8 | 0.3103 | 0.6049 | 0.4098 |
| pixelseal | >20% | 0.8729 | 0.2850 | 0.8798 | 0.5823 | 0.3/0.7 | 0.8809 | 0.5858 | 0.9020 |
| pixelseal | clean holes | 0.9089 | 0.2136 | 0.9181 | 0.5574 | 0.2/0.7 | 0.9185 | 0.5556 | 0.9308 |
| trustmark | ≤1% | 0.0000 | 0.0301 | 0.0234 | 0.4457 | 0.05/0.4 | 0.0172 | 0.4997 | 0.0458 |
| trustmark | 1–5% | 0.0608 | 0.1307 | 0.0720 | 0.5060 | 0.2/0.5 | 0.0728 | 0.5151 | 0.1340 |
| trustmark | 5–20% | 0.3556 | 0.1280 | 0.4122 | 0.6547 | 0.1/0.6 | 0.4289 | 0.6635 | 0.4907 |
| trustmark | >20% | 0.8763 | 0.2508 | 0.8826 | 0.5779 | 0.2/0.8 | 0.8826 | 0.5713 | 0.9043 |
| trustmark | clean holes | 0.9092 | 0.1609 | 0.9172 | 0.5360 | 0.2/0.3 | 0.9171 | 0.5423 | 0.9335 |

### Jedna polityka dla całych 170 obrazów

Tutaj progi maksymalizują **globalne oficjalne mIoU**, a następnie te same parametry stosujemy do wszystkich koszyków. Nadal jest to optymistyczna selekcja na całym OOF. Nie wybieramy polityki według nieznanego podczas inference rozmiaru GT.

| Metoda | Polityka | Globalne mIoU | FG ≤1% | FG 1–5% | FG 5–20% | FG >20% | FG holes | Clean holes |
|---|---|---|---|---|---|---|---|---|
| pixelseal | baseline_t05 | 0.571916 | 0.0000 | 0.0243 | 0.2950 | 0.8729 | 0.9089 | 0.2512 |
| pixelseal | global_threshold_oracle_miou | 0.572067 | 0.0000 | 0.0237 | 0.2932 | 0.8713 | 0.9073 | 0.2537 |
| pixelseal | global_hysteresis_oracle_miou | 0.572073 | 0.0000 | 0.0232 | 0.2951 | 0.8730 | 0.9085 | 0.2511 |
| trustmark | baseline_t05 | 0.578721 | 0.0000 | 0.0608 | 0.3556 | 0.8763 | 0.9092 | 0.2215 |
| trustmark | global_threshold_oracle_miou | 0.579026 | 0.0000 | 0.0606 | 0.3469 | 0.8736 | 0.9066 | 0.2275 |
| trustmark | global_hysteresis_oracle_miou | 0.579325 | 0.0000 | 0.0647 | 0.3544 | 0.8758 | 0.9085 | 0.2212 |

## 3. Zamrożony oficjalny dekoder i cropy

Wszystkie 43 małe PS i 45 TM. Pełny obraz trafia bezpośrednio do wrappera. Crop bbox/margines/tło: bilinear antialias do 256, potem niezmieniony wrapper. Natywna skala TM to 245; osobny bbox_native sprawdza bezpośredni resize. Losowy crop 128 tła wybierany wyłącznie spośród okien bez GT; kontrola 64 losowych wiadomości daje około 50%. GT-bbox jest diagnostycznym oracle, nie dostępną cechą testową.

| Metoda | Koszyk | n | Pełny | BBox→256 | 128→256 | 192→256 | Tło→256 | BBox ≥90%: n |
|---|---|---|---|---|---|---|---|---|
| pixelseal | ≤1% | 17 | 51.31% | 54.14% | 51.22% | 50.32% | 50.16% | 0 |
| pixelseal | 1–5% | 26 | 51.04% | 64.62% | 50.99% | 49.76% | 50.35% | 3 |
| trustmark | ≤1% | 9 | 50.00% | 67.78% | 50.44% | 49.00% | 51.33% | 2 |
| trustmark | 1–5% | 36 | 49.53% | 75.39% | 52.17% | 49.31% | 50.58% | 19 |

Okna 128/192 są wycentrowane na bbox, ale przy wydłużonym GT mogą ucinać znak: średni zachowany GT dla 1–5% to PS 83,7%/90,7%, TM 85,6%/93,1%. To pomiar ograniczenia tych okien, nie test pełnego cropu z marginesem dla każdego wydłużonego obiektu. Bezpośredni bbox→245 dla TM daje 68,22%/75,67%, wobec 67,78%/75,39% przez 256.

Dodatkowa kontrola po wyniku stałych okien: rozszerzenie każdej krawędzi bbox o 16 lub 32 px (ograniczone brzegiem obrazu), zachowujące 100% GT. Słaby wynik marginesu nie wynika więc wyłącznie z ucięcia watermarka.

| Metoda | Koszyk | Dokładny bbox | BBox + 16 px | BBox + 32 px |
|---|---|---|---|---|
| pixelseal | ≤1% | 54.14% | 49.40% | 49.68% |
| pixelseal | 1–5% | 64.62% | 50.17% | 49.01% |
| trustmark | ≤1% | 67.78% | 48.33% | 50.44% |
| trustmark | 1–5% | 75.39% | 50.11% | 49.89% |

| Metoda | Koszyk | Kształt GT | n | Bit accuracy bbox |
|---|---|---|---|---|
| pixelseal | ≤1% | fill>0,99 | 17 | 54.14% |
| pixelseal | 1–5% | fill>0,99 | 17 | 72.08% |
| pixelseal | 1–5% | fill≤0,99 | 9 | 50.52% |
| trustmark | ≤1% | fill>0,99 | 8 | 70.25% |
| trustmark | ≤1% | fill≤0,99 | 1 | 48.00% |
| trustmark | 1–5% | fill>0,99 | 19 | 99.05% |
| trustmark | 1–5% | fill≤0,99 | 17 | 48.94% |

Podział według fill-ratio jest analizą następczą po wyniku cropów. Odzysk wiadomości wskazuje na zgodną geometrię osadzenia. Brak odzysku nie dowodzi transplantacji: kontrolne lokalne osadzanie PS 48/64 również często nie daje pełnego odczytu.

### Tolerancja błędu położenia i rozmiaru cropu

Test następczy po słabym wyniku marginesów: te same 17 wariantów na każdym z 88 małych obrazów. Poniżej niemal prostokątne GT (fill>0,99). Przesunięcia to średnia czterech kierunków ±x/±y, z ograniczeniem do obrazu, nie najlepszy kierunek. BBox z siatki128 pochodzi z **idealnego GT128**, a nie predykcji modelu.

| Metoda | Koszyk | n | Dokładny bbox | GT128 bbox | Margines4 | Margines8 | Przesunięcie4 | Przesunięcie8 |
|---|---|---|---|---|---|---|---|---|
| pixelseal | ≤1% | 17 | 54.14% | 53.15% | 55.68% | 50.32% | 51.33% | 50.67% |
| pixelseal | 1–5% | 17 | 72.08% | 72.66% | 73.76% | 61.70% | 67.23% | 63.03% |
| trustmark | ≤1% | 8 | 70.25% | 74.88% | 77.88% | 52.38% | 63.72% | 59.09% |
| trustmark | 1–5% | 19 | 99.05% | 99.11% | 98.63% | 72.42% | 98.76% | 97.37% |

**TM ma praktyczną tolerancję przesunięcia:** w prostokątach 1–5% przesunięcie o 8 px nadal daje 97,37%, a bbox z GT128 99,11%. Silniejszy problem to rozmiar: margines4 daje 98,63%, margines8 już72,42%. Nie trzeba więc od razu zakładać subpikselowej lokalizacji. Pozostaje niewykonany test uzyskania takich propozycji bez GT. [Wyniki tolerancji PS](results/crop_tolerance/pixelseal/summary.json), [TM](results/crop_tolerance/trustmark/summary.json).

### Kontrole: lokalne osadzanie kontra fragment pełnego obrazu

16 tych samych źródeł COCO i wiadomości per metoda. Oficjalny embedder, domyślna siła, FP32 i dokładna kwantyzacja uint8. To nowy mały zestaw kontrolny, nie oryginalne realizacje z treningu na klastrze.

| Metoda | Bok cropu | Lokalne osadzenie | Fragment embed 512 | Czysty crop |
|---|---|---|---|---|
| pixelseal | 48 | 58.33% | 48.32% | 49.34% |
| pixelseal | 64 | 59.81% | 49.98% | 50.29% |
| pixelseal | 128 | 92.26% | 51.07% | 51.20% |
| pixelseal | 512 | 100.00% | 100.00% | 50.39% |
| trustmark | 48 | 83.19% | 48.75% | 48.75% |
| trustmark | 64 | 94.94% | 49.31% | 48.44% |
| trustmark | 128 | 99.63% | 50.00% | 52.25% |
| trustmark | 512 | 100.00% | 100.00% | 48.75% |

## 4. Fizyka sygnału po uint8

Δ mierzone w prostokącie osadzenia, względem identycznego czystego covera. Jednostki 0–255. Zero pixel: wszystkie trzy kanały mają Δ=0. Y/Cb/Cr to liniowe reszty BT.601, bez offsetów. Niski JND oznacza dolny kwartyl oryginalnej mapy na cropie embeddera; wyniki średnie po 16 coverach.

| Metoda | Bok | Obszar | Średnie |ΔRGB| | Piksele Δ=0 | |ΔY| | |ΔCb| | |ΔCr| |
|---|---|---|---|---|---|---|---|
| pixelseal | 64 | all | 0.6446 | 51.45% | 0.6455 | 0.0077 | 0.0038 |
| pixelseal | 64 | high_jnd_q75 | 1.1034 | 30.28% | 1.1054 | 0.0127 | 0.0065 |
| pixelseal | 64 | low_jnd_q25 | 0.2582 | 74.88% | 0.2591 | 0.0050 | 0.0021 |
| pixelseal | 64 | smooth_q25 | 0.5321 | 59.76% | 0.5325 | 0.0077 | 0.0035 |
| pixelseal | 512 | all | 0.9495 | 36.07% | 0.9500 | 0.0037 | 0.0028 |
| pixelseal | 512 | high_jnd_q75 | 1.5826 | 19.44% | 1.5831 | 0.0070 | 0.0060 |
| pixelseal | 512 | low_jnd_q25 | 0.4263 | 58.90% | 0.4268 | 0.0020 | 0.0014 |
| pixelseal | 512 | smooth_q25 | 0.5493 | 51.42% | 0.5492 | 0.0043 | 0.0046 |
| trustmark | 64 | all | 0.9123 | 28.14% | 0.8986 | 0.1187 | 0.0881 |
| trustmark | 64 | smooth_q25 | 0.8471 | 30.46% | 0.8369 | 0.1039 | 0.0819 |
| trustmark | 512 | all | 1.2653 | 23.07% | 1.2410 | 0.1603 | 0.1027 |
| trustmark | 512 | smooth_q25 | 1.2236 | 25.44% | 1.2091 | 0.1381 | 0.0931 |

Lokalny kod `unet_base_yuv_quant` wytwarza jednokanałowy sygnał dodawany do RGB, z JND i clippingiem. Zmierzone małe reszty chromy po uint8 oznaczają, że nie należy narzucać matematycznie zerowego ΔCb/Cr na końcowym obrazie. TrustMark też ma przewagę energii Y; sama przewaga Y nad chromą nie jest jeszcze sprawdzonym klasyfikatorem watermarka.

## 5. Geometria: dane realne i obecny generator

Maski realne 512 dla 340 obrazów. Generator: 1024 widoki na metodę, odtworzenie geometrii pierwszych 32 batchy po32 według obecnego kodu i RNG seed 42, z clean twins/complement/move. To **replay kodu**, nie odczyt archiwalnych masek treningowych. Pierwsze 24 grupy porównano z oryginalnym `pair_views` na siatce 128. Nie porównujemy do historycznego generatora starego ConvNeXt.

| Metoda | Zbiór | Polaryzacja | n | Fill bbox | Obwód/pole | Komponenty8 | Grubość px | Zgodność krawędzi |
|---|---|---|---|---|---|---|---|---|
| pixelseal | real | foreground | 170 | 0.8527 | 0.0446 | 1.8824 | 117.2467 | 0.5034 |
| pixelseal | real | minority | 170 | 0.6259 | 0.0770 | 2.6059 | 47.7220 | 0.5046 |
| pixelseal | generator_replay | foreground | 1024 | 0.8601 | 0.0743 | 4.9795 | 117.2251 | 0.4653 |
| pixelseal | generator_replay | minority | 1024 | 0.7277 | 0.1596 | 11.5586 | 48.2000 | 0.4669 |
| trustmark | real | foreground | 170 | 0.8496 | 0.0387 | 1.9059 | 112.8379 | 0.4994 |
| trustmark | real | minority | 170 | 0.6002 | 0.0738 | 2.7941 | 65.3037 | 0.4988 |
| trustmark | generator_replay | foreground | 1024 | 0.8649 | 0.0709 | 5.2480 | 116.9515 | 0.4448 |
| trustmark | generator_replay | minority | 1024 | 0.7356 | 0.1491 | 12.6230 | 54.1396 | 0.4436 |

Fill/thickness pomijają puste maski. Grubość to 2×EDT na lokalnych maksimach 3×3, przybliżenie szerokości fragmentów w pikselach 512. Zgodność krawędzi: odsetek wewnętrznej granicy maski w odległości ≤2 px od górnego kwintyla Sobela. **To proxy krawędzi obrazu, nie semantyczna adnotacja obiektu.** Replay używa 16 lokalnych coverów COCO, nie oryginalnych zdjęć klastra; interpretacja tej kolumny jest ograniczona. Pełne p10/mediana/p90 w JSON.

| Metoda | Zbiór | Koszyk | n | % widoków | Fill bbox | Komponenty | Grubość px |
|---|---|---|---|---|---|---|---|
| pixelseal | real | ≤1% | 17 | 10.0% | 1.0000 | 1.00 | 20.2 |
| pixelseal | real | 1–5% | 26 | 15.3% | 0.8627 | 1.27 | 43.5 |
| pixelseal | real | 5–20% | 30 | 17.6% | 0.6728 | 2.37 | 50.4 |
| pixelseal | generator_replay | ≤1% | 75 | 7.3% | 0.6551 | 5.56 | 16.3 |
| pixelseal | generator_replay | 1–5% | 167 | 16.3% | 0.7918 | 8.76 | 27.4 |
| pixelseal | generator_replay | 5–20% | 114 | 11.1% | 0.8101 | 12.60 | 70.0 |
| trustmark | real | ≤1% | 9 | 5.3% | 0.9451 | 1.00 | 29.1 |
| trustmark | real | 1–5% | 36 | 21.2% | 0.7921 | 1.97 | 53.6 |
| trustmark | real | 5–20% | 25 | 14.7% | 0.7400 | 1.92 | 88.8 |
| trustmark | generator_replay | ≤1% | 77 | 7.5% | 0.7991 | 4.82 | 20.5 |
| trustmark | generator_replay | 1–5% | 158 | 15.4% | 0.7829 | 12.03 | 42.0 |
| trustmark | generator_replay | 5–20% | 125 | 12.2% | 0.8284 | 11.44 | 89.6 |

**Nie obniżać globalnie fill-ratio do 0,40–0,65:** wszystkie 17 najmniejszych realnych PS mają pełny prostokąt (fill 1). Dla TM 1–5% niemal prostokątne 19/36 dają prawie idealny odczyt. Generator ma równocześnie zbyt wiele rozrzuconych komponentów (minority średnio 11,6/12,6 vs 2,6/2,8 real). Kształt trzeba dopasować warunkowo do reżimu, wielkości i polaryzacji. W replay udział ≤1% to 7,3%/7,5% wszystkich widoków, a nie deklarowane 20% minority źródeł; balansować po utworzeniu par i ostatecznych masek.

## 6. Kolejność dalszych ablacji

1. **Generator v2:** jawne embed rect i final mask; lokalne małe prostokąty oraz osobny tryb transplantacji z innego obrazu / clean holes z obiektu. Zachować 25–35% same-image mixing. Balans koszyków mierzyć na końcowych widokach. Porównać pojedynczą zmianę do aktualnego generatora przy identycznych 400 krokach pretrainingu i 150 adaptacji/fold.
2. **TM: dopasowany crop/refiner:** najpierw test kandydatów dostępnych bez GT. GT-bbox daje mocny sufit dla prostokątów; stałe 128/192 nie uzasadniają same z siebie pełnego sliding-window. Raportować recall propozycji dla pustych predykcji bazowych.
3. **PS: pomocniczy residual head w Y:** porównać L1 z kontrolą predykcji Δ=0; przy wielu zerach zwykłe średnie L1 może promować zerowy head. Sprawdzać residual error na niezerowych pikselach oraz właściwe realne FG IoU.
4. **Loss:** kontrolowana pojedyncza zmiana nadal może poprawić wyuczone cechy/recall; słaby oracle nie dowodzi, że retraining lossu nie pomoże. Nie oczekiwać dużego zysku z samego obniżenia progu.
5. **SRM/Bayar i semantyka:** oddzielne ablacje po powyższych. Aktualny model już ma lokalną gałąź 512 z pierwszym blokiem stride 1, dalszym wyjściem stride 4 i skipami piramidy. Zmiana filtrów to ablacja reprezentacji, a nie naprawa nieobecnego skipu. Brak dotąd testu przewagi semantyki dla PS.

Etapy 1–2 **nie zostały wytrenowane w tym zadaniu**. [Brief do dalszej pracy na A100](HANDOFF_A100.md).

## 7. Artefakty, koszt i ograniczenia

Suma czasów ostatnich wykonań skryptów wynosi około 166.5 s (bez pisania kodu, wcześniejszych prób, transferu i budowy dashboardu). Diagnostyka OOF 12.0 s; geometria 105.2 s. Maksymalna pamięć zaalokowana przez CUDA w testach dekodera i kontrolach pozostawała poniżej 0,5 GB, więc ten etap nie wymagał A100. Nie jest to pomiar kosztu treningu nowego modelu.

OOF jest wielokrotnie używanym zbiorem rozwojowym, nie nowym niezależnym testem. Oracle z GT służy diagnozie. Kontrole sygnału obejmują 16 źródeł, nie pełną pulę treningową. Oryginalne syntetyczne pary i ich manifesty pozostają na klastrze; dokładny audyt archiwalnych realizacji wymaga ich odczytu. Brak semantycznych adnotacji konkursowych ogranicza ocenę granic obiektów.

- [OOF i progi JSON](results/oof.json), [per-image CSV](results/oof_per_image.csv).
- [Geometria i rozkłady](results/geometry.json), [PS CSV](results/geometry_pixelseal.csv), [TM CSV](results/geometry_trustmark.csv).
- [Cropy PS](results/decoder/pixelseal/per_image.csv), [cropy TM](results/decoder/trustmark/per_image.csv).
- [Sygnał PS](results/signal/pixelseal/summary.json), [sygnał TM](results/signal/trustmark/summary.json).
- [Weryfikacja](results/verification.json), [pochodzenie i hashe](results/provenance.json).

Galerie wybierają najniższą i najwyższą bit accuracy bbox w każdym małym koszyku; nie są losową próbką.

![PixelSeal](results/gallery/example_pixelseal.png)

![TrustMark](results/gallery/example_trustmark.png)
