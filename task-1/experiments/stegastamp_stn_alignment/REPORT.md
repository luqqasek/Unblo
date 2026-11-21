# Eksperyment 1: pełna głowica StegaStamp po korekcie STN

**Status: aktywna baza do dalszej optymalizacji — decyzja użytkownika.** Przyjęto wariant z poprawioną geometrią STN mimo mieszanych wyników CV, ponieważ usuwa potwierdzony błąd współrzędnych. Kolejne warianty porównujemy z tym modelem na tych samych foldach i seedach. Obowiązujący protokół opisuje [indeks eksperymentów](../README.md).

**Wynik:** średnie OOF mIoU 0.46094 → 0.46512 (+0.00418). Poprawa w 1/3 seedach i 5/9 parach fold–seed. Dla operacyjnego seedu 42: 0.45621 → 0.45169 (-0.00452).

Korekta została zaimplementowana i pełna głowica została ponownie wytrenowana. Wynik jest mieszany: średnia poprawia się dzięki seedowi 123, seed 42 pogarsza się, a seed 7 jest niemal bez zmian. Warunkowy przedział bootstrap dla średniej różnicy obejmuje zero. Najlepszym zaakceptowanym zgłoszeniem pozostaje #1202; wybór nowej bazy rozwojowej nie zmienia tego wyniku ani domyślnego seedu 42.

Wytrenowano od zera 9 pełnych głowic na wyrównanych cechach: 3 identyczne zewnętrzne foldy × seedy 42/7/123. Bazą są zapisane modele stable_selection, użyte przez najlepszy wariant #1202. Wszystkie 9 bazowych checkpointów ponownie odtworzyło zapisane predykcje. Nie użyto korekty wiadomością z eksperymentu #1299.

Zmienia się wyłącznie geometria czterech map cech przed obecną klasą Head. Zastosowano odwrotność rzeczywistego STN z konwencją samplera 398/399, środkami convolutions i binów adaptive pooling, bilinear oraz border padding. Nowy cache zgadza się z cache sondy liniowej z tolerancją 1e-7. Nie przekształcano gotowych predykcji i nie wczytywano starych wag do nowego treningu.

Architektura, liczba parametrów, loss 0,5 BCE + 0,5 Dice, AdamW 0,001, weight decay 0,0001, batch 8, max 50 epok, patience 8 pozostają identyczne. Dobór epoki: pooled mIoU przy 0,5 z trzech stratyfikowanych wewnętrznych foldów. Próg: maksimum na tych samych wewnętrznych predykcjach w [0,3;0,4;0,5;0,6;0,7]. Następnie świeży trening na całym outer-train. Identyczny algorytm selekcji nie oznacza wymuszenia tej samej wybranej epoki i progu.

## Wyniki OOF według seedu

| Seed | Raw mIoU | Aligned mIoU | Zmiana | Raw AUC | Aligned AUC |
|---|---:|---:|---:|---:|---:|
| 42 | 0.456213 | 0.451694 | -0.004519 | 0.662973 | 0.656637 |
| 7 | 0.464327 | 0.464105 | -0.000222 | 0.653882 | 0.660162 |
| 123 | 0.462290 | 0.479566 | +0.017276 | 0.667157 | 0.692846 |

## Agregacja wszystkich seedów

| Metryka | Raw | Aligned | Zmiana |
|---|---:|---:|---:|
| miou | 0.460944 | 0.465122 | +0.004178 |
| auc | 0.661337 | 0.669882 | +0.008545 |
| ap | 0.565486 | 0.574932 | +0.009446 |
| fg_iou | 0.490565 | 0.486436 | -0.004129 |
| clean_iou | 0.431322 | 0.443808 | +0.012486 |
| boundary_f1 | 0.051300 | 0.060240 | +0.008940 |
| pred_area | 0.561590 | 0.551963 | -0.009626 |
| area_error | 0.104736 | 0.108610 | +0.003873 |

Poprawa mIoU w **5/9** parach fold–seed. OOF waży wszystkie obrazy równo (foldy mają 57/57/56 obrazów); seedy również równo. To 170 unikalnych obrazów, nie 510 niezależnych próbek. Warunkowy bootstrap parowanej różnicy mIoU po obrazach, po uśrednieniu seedów: [-0.001765; +0.010081].

## Wszystkie foldy i wybór parametrów

| Fold | Seed | Raw mIoU | Aligned mIoU | Delta | Epoki aligned | Próg aligned |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.468685 | 0.469989 | +0.001304 | 23 | 0.6 |
| 1 | 7 | 0.469747 | 0.482987 | +0.013241 | 20 | 0.6 |
| 1 | 123 | 0.473814 | 0.472791 | -0.001023 | 40 | 0.5 |
| 2 | 42 | 0.444979 | 0.433479 | -0.011500 | 19 | 0.5 |
| 2 | 7 | 0.449557 | 0.452938 | +0.003381 | 13 | 0.6 |
| 2 | 123 | 0.442874 | 0.477080 | +0.034207 | 27 | 0.6 |
| 3 | 42 | 0.454954 | 0.451614 | -0.003339 | 15 | 0.5 |
| 3 | 7 | 0.473843 | 0.456252 | -0.017592 | 28 | 0.4 |
| 3 | 123 | 0.470325 | 0.488992 | +0.018667 | 38 | 0.4 |

## Małe, średnie i duże obszary

| GT area | Unikalne obrazy | Raw FG IoU | Aligned FG IoU | Raw boundary F1 | Aligned boundary F1 |
|---|---:|---:|---:|---:|---:|
| small_le5pct | 45 | 0.015510 | 0.020241 | 0.023629 | 0.033156 |
| mid_5to50pct | 40 | 0.150961 | 0.154165 | 0.108949 | 0.106677 |
| large_gt50pct | 85 | 0.901879 | 0.889608 | 0.038821 | 0.052726 |

Boundary F1: granica wewnętrzna, bez ramki obrazu, tolerancja 2 komórki siatki 128 (8 px obrazu). AUC i AP są średnimi per obraz, nie pooled po pikselach.

## Przykłady przed i po

Dobór ustalony na podstawie poprzedniego audytu: najgorszy, środkowy i najlepszy raw mIoU w każdej grupie wielkości, seed42. Nie wybierano obrazów według poprawy nowego wariantu.

![Porównanie small](results/stegastamp_small_examples.png)
![Porównanie mid](results/stegastamp_mid_examples.png)
![Porównanie large](results/stegastamp_large_examples.png)

Po obejrzeniu 9 przykładów: misie (53934df0429a) nadal dostają niemal pełną maskę, a czysty prostokąt przy Big Benie (bb4b0481f38e) nadal znika. Żółty autobus (bd93c8feb303) poprawia mIoU około 0,739 → 0,783 i traci dodatkową fałszywą wyspę. Sygnalizator (d02e15fbddf3) poprawia 0,156 → 0,281, lecz pozostaje daleki od GT. Tort (de11c61674be) pogarsza 0,534 → 0,430 przez większe fałszywe obszary, a zegar (42ee90cc410d) 0,617 → 0,466 przez zanik czystego wycięcia. Poprawa geometrii nie usuwa dominujących problemów lokalizacji.


## Ograniczenia i artefakty

To porównanie rozwojowe na wielokrotnie analizowanych foldach. OOF etykiety służą wyłącznie ocenie, a nie wyborowi epoki/progu; nie jest to nowy nietknięty test. Nie wybieramy najlepszego seedu. Interpolacja również wygładza cechy, więc bez osobnej kontroli wygładzania nie można przypisać całej różnicy wyłącznie geometrii. Nie oceniano testu konkursowego ani leaderboardu; poprzednie checkpointy i zgłoszenia zachowano.

[Podsumowanie JSON](results/summary.json), [metryki per obraz](results/per_image_metrics.csv), [wyniki według wielkości](results/size_stratified_metrics.csv), [liczebności problemów](results/failure_counts.csv), [weryfikacja](results/verification.json).

Odtworzenie z katalogu repozytorium:
```powershell
task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stn_alignment/run.py
task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stn_alignment/report.py
task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stn_alignment/verify.py
task-1/.venv/Scripts/python.exe task-1/dashboard/build.py
```
