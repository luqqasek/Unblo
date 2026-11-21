# Syntetyki dla trzech rodzin i kontrola wielkości danych

Plan zapisany przed oceną nowych wariantów. Kontynuacja `synthetic_pretrain_2k`.

## Hipotezy i porównania

1. Czy prawdziwe syntetyki PixelSeal i TrustMark poprawiają ich niezmienione głowice?
2. Czy 4000 różnych coverów poprawia wyniki bardziej niż 2000?
3. Czy zysk wynika z większej różnorodności, czy większej liczby kroków?

| Wariant | Syntetyki treningowe | Epoki pretrain | Kroki batch8 | Znaczenie |
|---|---:|---:|---:|---|
| real_only | 0 | 0 | 0 | Dopasowana kontrola |
| n2k_s2500 | 2000 | 10 | 2500 | Dotychczasowy budżet |
| n4k_s2500 | 4000 | 5 | 2500 | Więcej źródeł przy stałym budżecie |
| n2k_s5000 | 2000 | 20 | 5000 | Kontrola dłuższego treningu |
| n4k_s5000 | 4000 | 10 | 5000 | Więcej źródeł i więcej kroków |

Każdy wariant: trzy seedy 42/7/123, trzy istniejące outer-foldy i późniejszy fine-tuning wyłącznie na realnym outer-train. Pełna siatka, bez wybierania wariantów na pierwszym foldzie. PS/TM zachowują istniejący inner80/20, SS trzy inner-foldy. Epoka 1–50, patience8 i próg [0.3,0.4,0.5,0.6,0.7] wybrane tylko wewnętrznie; świeży refit od właściwego checkpointu synthetic. AdamW .001/.0001, batch8, 0.5 BCE + 0.5 Dice. SS zawsze inverse-STN. Nie zmieniamy architektury, loss ani protokołu fine-tuningu jednocześnie z danymi.

Przed fine-tuningiem mierzymy też synthetic-only na tych samych real170, przy stałym progu0.5; jest to pomocnicza analiza transferu, a nie zamiennik głównego porównania po fine-tuningu. Syntetyczne checkpointy po 10/20 (2k) i 5/10 (4k) powstają w jednej ciągłej trajektorii per seed/rozmiar; zapis optymalizatora umożliwia dokładne wznowienie.

## Dane i geometria

COCO val2017 z poprzednio pobranych oficjalnych archiwów. Te same źródła i generator: 45% lokalnego embed(crop), 45% pełnego embed z czystym otworem, 5% pełnych i 5% pustych masek. Prostokąty, elipsy i obiekty COCO. Bez zmiany siły znakowania, JPEG lub generatywnych edycji.

Zachowujemy wcześniejsze 2000 źródeł treningowych i 200 syntetycznej walidacji. Dodatkowe 2000 źródeł to dalsze pozycje tej samej deterministycznej permutacji COCO, z pominięciem walidacji. Logiczne indeksy: mały train0:2000, holdout2000:2200, większy train=0:2000 + 2200:4200. Ani obrazy, ani maski real-validation nie są materiałem generatora. Dokładne SHA i dHash służą kontroli powtórzeń; nie gwarantują pełnego rozpoznania wspólnego źródła po mocnej edycji.

StegaStamp wykorzystuje bez zmian już zweryfikowane 2200 syntetyków; powstaje dodatkowe2000. PS/TM używają własnych oficjalnych encoderów (PixelSeal256bit, TrustMark Q100bit bezECC). Dla TrustMark prostokąt przekazywany do encodera jest rozszerzany do proporcji <=2:1, żeby uniknąć centralnego znakowania części patcha. Po encode przywracamy czyste piksele poza docelową maską; zapisujemy zarówno maskę, jak i rzeczywisty bbox encodera. Tym samym cienka maska może leżeć wewnątrz szerszego poprawnie oznakowanego cropa.

Wszystkie mapy cech float32, natywne poziomy i poolingcap32 jak w bazie. Dane duże przechowywane w plikach NPY/memmap; batch jest przenoszony na GPU bez kwantyzacji. Zamrożone ekstraktory eval. Stałe urządzenie per metoda dla kontroli i wariantów; TF32 off, deterministyczne algorytmy. Nie nadpisujemy wcześniejszych artefaktów.

## Ocena

Oficjalne per-image mIoU, foreground i clean IoU, macro AUC/AP, tie-aware Top-K jako oracle-area diagnostic, wielkość maski, małe regiony<=5%, czyste otwory/duże regiony. Główna analiza: parowane różnice po obrazach, uśredniając trzy seedy; bootstrap95% CI (5000 powtórzeń), osobno każdy seed i fold. Cztery warianty badane na wielokrotnie używanym zbiorze rozwojowym; CI warunkowe, bez korekty wielokrotnych porównań i bez ponownego uczenia bootstrapów.

Priorytetowe kontrasty: 2k2500-real_only, 4k2500-2k2500, 4k5000-2k5000, 2k5000-2k2500, 4k5000-2k2500. Nie traktować wzrostu mIoU przy spadku foreground IoU jako rozwiązania lokalizacji. Obecne zgłoszenie #1496 pozostaje zachowane; ten eksperyment nie używa leaderboardu do strojenia.
