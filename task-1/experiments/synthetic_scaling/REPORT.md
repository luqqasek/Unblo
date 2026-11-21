# Syntetyki dla wszystkich rodzin: 2k vs 4k

**Wykonano cały zaplanowany eksperyment: 3 metody × 5 wariantów × 3 seedy × 3 foldy = 135 końcowych fitów OOF.** Wszystkie poniższe główne wyniki dotyczą modeli po fine-tuningu na realnych danych. 510 unikalnych obrazów, po 170 na metodę; powtórzenia seedów nie zwiększają liczby niezależnych zdjęć.

## Odpowiedź na pytania

Pretraining na 2k syntetyków poprawia wynik wszystkich trzech rodzin. Dalsze podwojenie puli ma dodatnie przedziały różnicy dla PixelSeal przy obu budżetach. Dla TrustMark przewaga 4k pozostaje niepewna. StegaStamp korzysta przede wszystkim z większej liczby kroków na dotychczasowych danych. Są to wyniki rozwojowego CV; małe znaki wodne nadal pozostają trudne.

W tabeli liczba kroków dotyczy pretrainingu. Potem każdy wariant przechodzi realny fine-tuning z tym samym protokołem wewnętrznego doboru epoki i progu. Liczby są średnimi oddzielnych przebiegów trzech seedów.

| Metoda | Tylko real | 2k / 2500 kroków | 4k / 2500 kroków | 2k / 5000 kroków | 4k / 5000 kroków |
|---|---:|---:|---:|---:|---:|
| pixelseal | 0.51131 | 0.54365 | 0.55360 | 0.54257 | 0.55081 |
| trustmark | 0.49082 | 0.53713 | 0.54247 | 0.53891 | 0.54407 |
| stegastamp | 0.46129 | 0.50719 | 0.50692 | 0.51789 | 0.51720 |

Główna miara: oficjalna średnia per-image dwuklasowego mIoU. Różne rodziny mają odrębne modele; nie porównujemy ich jako jednego modelu.

- **pixelseal:** 2k względem real-only +0.03233. Podwojenie liczby danych przy 2500 krokach: +0.00995, przy 5000: +0.00824. Praktyczne 2k×10epok → 4k×10epok: +0.00716.
- **trustmark:** 2k względem real-only +0.04631. Podwojenie liczby danych przy 2500 krokach: +0.00534, przy 5000: +0.00516. Praktyczne 2k×10epok → 4k×10epok: +0.00694.
- **stegastamp:** 2k względem real-only +0.04590. Podwojenie liczby danych przy 2500 krokach: -0.00026, przy 5000: -0.00069. Praktyczne 2k×10epok → 4k×10epok: +0.01001.

Sama dodatnia różnica punktowa nie rozstrzyga przewagi. Poniżej parowane przedziały i wyniki seedów, oraz oddzielenie większej puli od dłuższego treningu.

## Co dokładnie porównano

Pula 4000 zawiera wcześniejsze 2000 przykładów i 2000 nowych zdjęć. Stałe 200 coverów syntetycznej walidacji jest wyłączone z obu treningów. Wszystkie źródła COCO pochodzą z jednej ustalonej permutacji; po jednym syntetyku na źródło i metodę. StegaStamp zachowuje dokładnie te same wcześniejsze przykłady oraz inverse-STN. PixelSeal: 256 bitów; TrustMark Q: 100 bitów bez ECC. Użyto rzeczywistych encoderów ze starter kitu.

Pretraining: 2k × 10 epok i 4k × 5 epok mają po 2500 aktualizacji; 2k × 20 epok i 4k × 10 epok mają po 5000. Przy tej samej liczbie kroków i batch size 8 model przetwarza tyle samo przykładów, ale inną liczbę unikalnych zdjęć. Cztery checkpointy są z góry określone; syntetyczna walidacja ich nie wybiera.

Fine-tuning: niezmieniona głowica i zamrożone ekstraktory; tylko realny outer-train. Istniejący podział wewnętrzny 80/20 dla PixelSeal i TrustMark lub trzy wewnętrzne foldy StegaStamp wybierają epokę i próg. Następnie następuje świeży refit od właściwego syntetycznego checkpointu. AdamW: learning rate 0.001, weight decay 0.0001; batch size 8; loss 0.5 BCE + 0.5 Dice; maks. 50 epok, patience 8; progi 0.3–0.7. Kontrole real-only są uczone tym samym kodem i na tym samym urządzeniu co warianty danej metody.

Dla TrustMark bbox przekazywany do encodera ma proporcje <=2:1. Cienka docelowa maska może znajdować się wewnątrz szerszego poprawnie oznakowanego cropa; poza maską przywracamy identyczne czyste piksele. Nie zmieniono architektury ani loss razem z liczebnością danych.

[Plan przed wynikami](README.md) · [Instrukcja odtworzenia](REPRODUCE.md). Badanie nie uruchamia predykcji testowych ani zgłoszeń; istniejący najlepszy wynik #1496 pozostaje zachowany.

### Kontrola generatora

Liczby 2k i 4k oznaczają przykłady treningowe na każdą metodę. Metody korzystają z tych samych zdjęć źródłowych: 4000 treningowych i 200 walidacyjnych coverów, a po zastosowaniu trzech encoderów powstało 12 600 par obraz–maska. Nie traktujemy ich jako 12 600 niezależnych zdjęć źródłowych.

![Przykładowe syntetyki i rzeczywisty residual](results/generation_examples.png)

Ten sam cover i maska, trzy rzeczywiste encodery. Kolumna residualu pokazuje średnią bezwzględną zmianę RGB, z osobną skalą każdego panelu; nie służy porównaniu siły znakowania.

Dla dziewięciu zapisanych przykładów z dodatkowej puli ponownie wykonano ekstrakcję na CPU: wszystkie 36 map zgadza się z cache’em w zadanej tolerancji. [Wynik kontroli](results/feature_replay_verification.json).

![Rozkład powierzchni masek](results/generation_areas.png)

Rozkład powierzchni porównano dla obu pul i realnych obrazów. Synteza zawiera po 5% całkowicie czystych i całkowicie oznakowanych przykładów; realne 170 obrazów każdej metody nie zawiera takich skrajnych masek. To zamierzona kontrola treningowa, ale również różnica domen. Pula 4k ma podobny rozkład powierzchni do 2k. [Liczby](results/generation_area_summary.json) · [Kontrola wszystkich masek i cech](results/data_verification.json).

## Efekt danych i efekt budżetu

| Metoda | Kontrast | Delta mIoU | CI95 | Dodatnie pary fold–seed |
|---|---|---:|---|---:|
| pixelseal | 2k2500 − real | +0.03233 | [+0.02200, +0.04289] | 9/9 |
| pixelseal | 4k2500 − 2k2500: dane | +0.00995 | [+0.00437, +0.01537] | 7/9 |
| pixelseal | 4k5000 − 2k5000: dane | +0.00824 | [+0.00169, +0.01502] | 8/9 |
| pixelseal | 2k5000 − 2k2500: kroki | -0.00108 | [-0.00479, +0.00258] | 6/9 |
| pixelseal | 4k5000 − 4k2500: kroki | -0.00279 | [-0.00755, +0.00201] | 2/9 |
| pixelseal | 4k5000 − 2k2500: oba | +0.00716 | [+0.00108, +0.01345] | 8/9 |
| trustmark | 2k2500 − real | +0.04631 | [+0.03502, +0.05791] | 9/9 |
| trustmark | 4k2500 − 2k2500: dane | +0.00534 | [-0.00162, +0.01227] | 6/9 |
| trustmark | 4k5000 − 2k5000: dane | +0.00516 | [-0.00034, +0.01075] | 6/9 |
| trustmark | 2k5000 − 2k2500: kroki | +0.00178 | [-0.00428, +0.00762] | 6/9 |
| trustmark | 4k5000 − 4k2500: kroki | +0.00160 | [-0.00266, +0.00573] | 6/9 |
| trustmark | 4k5000 − 2k2500: oba | +0.00694 | [-0.00004, +0.01367] | 7/9 |
| stegastamp | 2k2500 − real | +0.04590 | [+0.03495, +0.05725] | 9/9 |
| stegastamp | 4k2500 − 2k2500: dane | -0.00026 | [-0.00513, +0.00443] | 3/9 |
| stegastamp | 4k5000 − 2k5000: dane | -0.00069 | [-0.00533, +0.00398] | 5/9 |
| stegastamp | 2k5000 − 2k2500: kroki | +0.01070 | [+0.00593, +0.01552] | 5/9 |
| stegastamp | 4k5000 − 4k2500: kroki | +0.01028 | [+0.00584, +0.01481] | 7/9 |
| stegastamp | 4k5000 − 2k2500: oba | +0.01001 | [+0.00416, +0.01569] | 6/9 |

CI: 5000 bootstrapów 170 obrazów danej metody po uśrednieniu seedów, z parowaniem obrazów. Nie obejmują ponownego treningu ani zmienności innych pul syntetycznych. Zbiór rozwojowy był wcześniej wielokrotnie analizowany; wiele kontrastów bez korekty wielokrotnych porównań. Dodatnie foldy nie są niezależnymi testami.

![Parowane efekty danych i kroków](results/paired_effects.png)

![Skalowanie po fine-tuningu](results/scaling_curves.png)

## Czy poprawia się lokalizacja?

| Metoda | Wariant | FG IoU | Clean IoU | AUC macro | AP macro | Małe FG IoU | Małe AP | Małe delta vs empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pixelseal | Tylko real | 0.5177 | 0.5049 | 0.7136 | 0.6306 | 0.0398 | 0.0767 | +0.0032 |
| pixelseal | 2k / 2500 kroków | 0.5259 | 0.5614 | 0.7555 | 0.6456 | 0.0274 | 0.0772 | +0.0018 |
| pixelseal | 4k / 2500 kroków | 0.5390 | 0.5682 | 0.7747 | 0.6581 | 0.0338 | 0.0892 | +0.0043 |
| pixelseal | 2k / 5000 kroków | 0.5243 | 0.5609 | 0.7430 | 0.6369 | 0.0261 | 0.0719 | +0.0020 |
| pixelseal | 4k / 5000 kroków | 0.5335 | 0.5681 | 0.7581 | 0.6459 | 0.0286 | 0.0838 | -0.0018 |
| trustmark | Tylko real | 0.4989 | 0.4827 | 0.6823 | 0.5986 | 0.0225 | 0.0693 | -0.0138 |
| trustmark | 2k / 2500 kroków | 0.5318 | 0.5425 | 0.7501 | 0.6352 | 0.0479 | 0.1017 | +0.0044 |
| trustmark | 4k / 2500 kroków | 0.5254 | 0.5595 | 0.7491 | 0.6349 | 0.0319 | 0.0950 | +0.0064 |
| trustmark | 2k / 5000 kroków | 0.5337 | 0.5441 | 0.7521 | 0.6344 | 0.0370 | 0.1010 | -0.0002 |
| trustmark | 4k / 5000 kroków | 0.5352 | 0.5530 | 0.7507 | 0.6385 | 0.0371 | 0.0971 | +0.0037 |
| stegastamp | Tylko real | 0.4883 | 0.4343 | 0.6713 | 0.5745 | 0.0179 | 0.0474 | -0.0630 |
| stegastamp | 2k / 2500 kroków | 0.4806 | 0.5337 | 0.7435 | 0.6022 | 0.0166 | 0.0615 | -0.0108 |
| stegastamp | 4k / 2500 kroków | 0.4924 | 0.5214 | 0.7449 | 0.6034 | 0.0190 | 0.0649 | -0.0119 |
| stegastamp | 2k / 5000 kroków | 0.4941 | 0.5417 | 0.7519 | 0.6031 | 0.0244 | 0.0701 | -0.0031 |
| stegastamp | 4k / 5000 kroków | 0.4994 | 0.5350 | 0.7623 | 0.6073 | 0.0201 | 0.0615 | -0.0080 |

Małe maski: GT ≤ 5%; liczba unikalnych obrazów — pixelseal: 43, trustmark: 45, stegastamp: 45. AUC/AP są średnimi per obraz, nie pooled. AP to Average Precision. Top-K w CSV używa znanej powierzchni GT i jest tylko diagnostyką. Jeśli mIoU rośnie głównie przez clean IoU, nie oznacza to rozwiązania lokalizacji watermarku.

![Małe regiony](results/small_regions.png)

[Pełne przedziały powierzchni](results/size_stratified_metrics.csv) · [Tryb znakowania](results/embedding_mode_metrics.csv) · [Wszystkie obrazy](results/per_image_metrics.csv) · [Galeria błędów/popraw](results/gallery/index.html). Biny z n<10 oznaczono jako mało wiarygodne.

## Seedy i fine-tuning

| Metoda | Wariant | Seed42 | Seed7 | Seed123 | SD seedów |
|---|---|---:|---:|---:|---:|
| pixelseal | Tylko real | 0.51016 | 0.51233 | 0.51146 | 0.00109 |
| pixelseal | 2k / 2500 kroków | 0.54862 | 0.54170 | 0.54063 | 0.00434 |
| pixelseal | 4k / 2500 kroków | 0.54527 | 0.55745 | 0.55808 | 0.00722 |
| pixelseal | 2k / 5000 kroków | 0.54545 | 0.53976 | 0.54250 | 0.00285 |
| pixelseal | 4k / 5000 kroków | 0.55607 | 0.54802 | 0.54833 | 0.00456 |
| trustmark | Tylko real | 0.48855 | 0.51508 | 0.46882 | 0.02321 |
| trustmark | 2k / 2500 kroków | 0.53966 | 0.52782 | 0.54390 | 0.00834 |
| trustmark | 4k / 2500 kroków | 0.54018 | 0.54524 | 0.54198 | 0.00256 |
| trustmark | 2k / 5000 kroków | 0.52046 | 0.54459 | 0.55167 | 0.01636 |
| trustmark | 4k / 5000 kroków | 0.53820 | 0.54316 | 0.55084 | 0.00637 |
| stegastamp | Tylko real | 0.44441 | 0.46441 | 0.47505 | 0.01555 |
| stegastamp | 2k / 2500 kroków | 0.51514 | 0.50436 | 0.50206 | 0.00698 |
| stegastamp | 4k / 2500 kroków | 0.51200 | 0.50220 | 0.50658 | 0.00491 |
| stegastamp | 2k / 5000 kroków | 0.51366 | 0.52334 | 0.51665 | 0.00496 |
| stegastamp | 4k / 5000 kroków | 0.51624 | 0.51577 | 0.51959 | 0.00209 |

Pomocnicza kontrola synthetic-only używa stałego progu 0.5 na 170 realnych obrazach, bez uczenia na real. Fine-tuning ma próg dobrany w inner CV; różnica obejmuje więc adaptację i kalibrację. Główne decyzje w tym raporcie odnoszą się do modeli po fine-tuningu, zgodnie z wyborem użytkownika.

| Metoda | Wariant | Synthetic-only, próg 0.5 | Po fine-tuningu, próg 0.5 | Po fine-tuningu, próg inner CV |
|---|---|---:|---:|---:|
| pixelseal | 2k / 2500 kroków | 0.53901 | 0.54398 | 0.54365 |
| pixelseal | 4k / 2500 kroków | 0.55096 | 0.55549 | 0.55360 |
| pixelseal | 2k / 5000 kroków | 0.53150 | 0.54199 | 0.54257 |
| pixelseal | 4k / 5000 kroków | 0.55272 | 0.55121 | 0.55081 |
| trustmark | 2k / 2500 kroków | 0.52284 | 0.53561 | 0.53713 |
| trustmark | 4k / 2500 kroków | 0.53004 | 0.54376 | 0.54247 |
| trustmark | 2k / 5000 kroków | 0.50683 | 0.53959 | 0.53891 |
| trustmark | 4k / 5000 kroków | 0.51798 | 0.54495 | 0.54407 |
| stegastamp | 2k / 2500 kroków | 0.51153 | 0.50882 | 0.50719 |
| stegastamp | 4k / 2500 kroków | 0.51399 | 0.50697 | 0.50692 |
| stegastamp | 2k / 5000 kroków | 0.51641 | 0.51630 | 0.51789 |
| stegastamp | 4k / 5000 kroków | 0.51646 | 0.51565 | 0.51720 |

![Uczenie syntetyczne](results/pretrain_learning.png)

## Wiarygodność i ograniczenia

Kontrole obejmują zgodność cech z bazą, dokładnie zachowany dotychczasowy zbiór StegaStamp, poprawny bbox TrustMark, rozłączne źródła train/holdout, SHA/dHash względem 510 realnych obrazów, niezmienność istniejących modeli i zgłoszenia, oficjalny downsampling masek, mapowanie identyfikatorów OOF, wewnętrzny wybór parametrów i odtworzenie predykcji z checkpointów. Różnice CPU/GPU cech są zapisane w preflightach; nie nazywamy ich idealną zgodnością bitową.

Jeden zewnętrzny zbiór i jedna deterministyczna pula generatora. Dwukrotne zwiększenie do 4k nie odpowiada na pytanie o 10k/100k i nie dowodzi monotonicznej poprawy. Syntetyki nie odtwarzają wszystkich generatywnych edycji konkursowych; maski pochodzą z mieszania ze znanym czystym zdjęciem. Brak identyfikatorów źródeł real ogranicza pewność kontroli near-duplicates.

## Wniosek dla dalszych eksperymentów

- **pixelseal:** Większa pula poprawia wynik przy obu stałych budżetach; jest uzasadnienie do następnego, większego poziomu danych. Najwyższa średnia w tej siatce: 4k / 2500 kroków (0.55360); jest to wybór po analizie rozwojowej, nie niezależnie potwierdzony zwycięzca.
- **trustmark:** Efekt większej puli jest zależny od budżetu lub niepewny; przed dalszym skalowaniem sprawdzić stabilność i foreground metrics. Najwyższa średnia w tej siatce: 4k / 5000 kroków (0.54407); jest to wybór po analizie rozwojowej, nie niezależnie potwierdzony zwycięzca.
- **stegastamp:** Nie potwierdzono korzyści z większej puli w oficjalnym mIoU przy żadnym stałym budżecie. Natomiast wydłużenie treningu tych samych 2k przykładów poprawia mIoU o +0.01070, CI95 [+0.00593, +0.01552]. Wzrost w porównaniu 2k×10 → 4k×10 należy więc przypisywać przede wszystkim dodatkowym krokom, a nie wykazanemu efektowi liczebności. Osobne zmiany rankingu i obu klas pozostają widoczne w tabeli metryk. Najwyższa średnia w tej siatce: 2k / 5000 kroków (0.51789); jest to wybór po analizie rozwojowej, nie niezależnie potwierdzony zwycięzca.

Nie przenosić automatycznie jednej receptury na wszystkie rodziny. Przed wdrożeniem sprawdzić, czy zwycięski wariant poprawia ranking i regiony, czy tylko przewiduje więcej czystego tła. Nie zmieniać modelu ani generatora jednocześnie z kolejnym poziomem liczebności.

**Najbardziej uzasadniony kolejny test liczebności:** 8k syntetyków PixelSeal przy 5000 krokach, z dopasowaną kontrolą 4k przy 5000 krokach i identycznym fine-tuningiem. Poparciem są dodatnie różnice 2k → 4k przy obu dotychczasowych budżetach. Brak dalszego wzrostu mIoU/AP lub pogorszenie małych regionów przemawiałoby za zmianą generatora zamiast dalszego zwiększania liczby przykładów. Nie wykonano jeszcze tego testu i nie prognozujemy liczbowej poprawy.

Przy zwiększeniu puli dochodzą jednocześnie nowe covery, maski i wiadomości. Ten eksperyment oddziela dodatkowe przykłady od dodatkowych kroków, ale nie rozstrzyga, czy tę samą korzyść dałoby więcej wariantów maski na dotychczasowych coverach.

[Wyniki JSON](results/summary.json) · [Metryki CSV](results/summary_metrics.csv) · [Foldy](results/fold_metrics.csv) · [Koszt obliczeń](results/compute_cost.json) · [Weryfikacja](results/verification.json).
