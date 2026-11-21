# Diagnoza StegaStamp: dane, seed i wybór parametrów

## Zakres i wiarygodność

Odtworzono 510 zapisanych predykcji OOF z checkpointów oraz oficjalną metrykę (`frozen_extractor/verify_cv.py`). StegaStamp: średnia foldów 0,410596; odchylenie standardowe próby między foldami 0,062998. To zmienność podziałów, a nie przedział ufności ani pomiar zmienności seedów.

Użyto wyłącznie oryginalnego train i zamrożonych cache cech. Identyczne podziały zewnętrzne i wewnętrzne jak w poprzednim 3-fold CV. Seed 42 to oryginalny przebieg; 7 i 123 to nowe treningi. Każdy nowy seed sprawdzono przy stałej liczbie epok i progu z oryginalnego przebiegu oraz po ponownym doborze epoki i progu na wewnętrznej walidacji. Zmieniono jednocześnie inicjalizację i kolejność batchy; ten eksperyment nie rozdziela tych dwóch składników losowości. Nie użyto zbioru testowego ani leaderboardu. Wyniki są diagnostyczne na już analizowanych foldach.

## Zawartość foldów i rodzaj błędów

| Fold | mIoU | Udział GT | Udział predykcji | FP / obraz | FN / obraz | Małe regiony <25% | full_watermark=True |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4656 | 55.6% | 61.4% | 10.0% | 4.1% | 22/57 | 56.1% |
| 2 | 0.3418 | 46.7% | 80.1% | 34.3% | 0.8% | 30/57 | 47.4% |
| 3 | 0.4244 | 45.3% | 30.1% | 3.2% | 18.4% | 29/56 | 46.4% |

FP/FN to odsetek wszystkich komórek obrazu, nie precision/recall.

| Fold | mIoU częściowego watermarku (False) | mIoU pełnego watermarku z przywróconym fragmentem (True) | Pole otworów | Krawędzie / piksel | Trafność bitów | Maska przestrzenna |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.4581 (n=25) | 0.4713 (n=32) | 1.47% | 0.0140 | 0.7856 | 0.2611 |
| 2 | 0.2344 (n=30) | 0.4612 (n=27) | 1.09% | 0.0148 | 0.7351 | 0.2905 |
| 3 | 0.4636 (n=30) | 0.3792 (n=26) | 0.99% | 0.0117 | 0.7180 | 0.2817 |

Dostępna etykieta `full_watermark` odróżnia pełne oznakowanie z przywróconym czystym fragmentem od częściowego oznakowania. Metadane nie zawierają osobnych etykiet outpainting / usunięcie / dodanie / transplantacja, więc nie przypisujemy tych typów na podstawie samego kształtu maski. Otwory obliczono przez binary_fill_holes na siatce 128×128, a krawędzie przez różnice sąsiednich komórek.

## Powtórzenia na tych samych podziałach

| Fold | Seed | Procedura | Epoki | Próg | mIoU |
|---|---:|---|---:|---:|---:|
| 1 | 42 | original | 22 | 0.4 | 0.4656 |
| 1 | 7 | fixed | 22 | 0.4 | 0.4647 |
| 1 | 7 | nested | 38 | 0.3 | 0.4674 |
| 1 | 123 | fixed | 22 | 0.4 | 0.4656 |
| 1 | 123 | nested | 15 | 0.5 | 0.4605 |
| 2 | 42 | original | 10 | 0.5 | 0.3418 |
| 2 | 7 | fixed | 10 | 0.5 | 0.4123 |
| 2 | 7 | nested | 18 | 0.5 | 0.4604 |
| 2 | 123 | fixed | 10 | 0.5 | 0.4022 |
| 2 | 123 | nested | 19 | 0.6 | 0.4485 |
| 3 | 42 | original | 12 | 0.5 | 0.4244 |
| 3 | 7 | fixed | 12 | 0.5 | 0.4048 |
| 3 | 7 | nested | 16 | 0.6 | 0.4566 |
| 3 | 123 | fixed | 12 | 0.5 | 0.4732 |
| 3 | 123 | nested | 13 | 0.6 | 0.4842 |

`fixed`: niezmienione epoki i próg z seed=42; `nested`: ponowny dobór na tym samym wewnętrznym podziale. `original`: oryginalny seed=42. Stałe epoki izolują zmienność treningu od zmienności doboru hiperparametrów.

| Fold | Średnia 3 seedów, stałe parametry | SD seedów | Średnia 3 seedów, dobór wewnętrzny | SD seedów |
|---|---:|---:|---:|---:|
| 1 | 0.4653 | 0.0005 | 0.4645 | 0.0036 |
| 2 | 0.3854 | 0.0381 | 0.4169 | 0.0653 |
| 3 | 0.4341 | 0.0352 | 0.4551 | 0.0299 |

Po przeważeniu obu kategorii full_watermark do wspólnych proporcji wszystkich 170 obrazów, mIoU foldów wynosi: 0.4647, 0.3478, 0.4214. To opisowa korekta składu zbioru, nie dowód przyczynowości; modele w foldach trenowano na innych danych.

## Odtworzenie

Z katalogu głównego repozytorium:
```powershell
task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_diagnosis/run.py
task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_diagnosis/report.py
python task-1/dashboard/build.py
```

Artefakty: `results/metrics.json`, `per_image.csv`, checkpointy i prawdopodobieństwa nowych przebiegów nested, po 6 przykładów na fold oraz wykres rozrzutu. Hipotezy i interpretacja: [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md).
