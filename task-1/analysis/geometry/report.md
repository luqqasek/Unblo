# Etap 0 — geometria ground truth

510 masek treningowych. Wyniki mIoU i bboxy liczone na oficjalnej siatce 128 × 128. Każdy obraz ma równą wagę. Wszystkie maski 512 × 512 po oficjalnym downsamplingu zgadzają się z masks_128.npz.

BBox to najmniejszy prostokąt osiowy obejmujący wszystkie jedynki GT, z włączonymi skrajnymi komórkami. Korzysta z ground truth: nie jest to wynik detektora ani maksimum po wszystkich prostokątach. Puste maski: predykcja bbox jest pusta.

| Grupa | N | Cały obraz: mIoU | BBox GT: mIoU | Pokrycie 128: średnia | Wypełnienie bbox: średnia | BBox idealny | BBox ≥ 0,95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 510 | 0.2482 | 0.6625 | 49.64% | 85.06% | 80 | 131 |
| pixelseal | 170 | 0.2470 | 0.6652 | 49.39% | 85.47% | 28 | 43 |
| pixelseal/full_watermark=True | 85 | 0.4486 | 0.4519 | 89.72% | 89.86% | 0 | 0 |
| pixelseal/full_watermark=False | 85 | 0.0453 | 0.8786 | 9.07% | 81.08% | 28 | 43 |
| trustmark | 170 | 0.2516 | 0.6621 | 50.33% | 85.14% | 28 | 44 |
| trustmark/full_watermark=True | 85 | 0.4540 | 0.4600 | 90.79% | 91.04% | 0 | 0 |
| trustmark/full_watermark=False | 85 | 0.0493 | 0.8641 | 9.86% | 79.24% | 28 | 44 |
| stegastamp | 170 | 0.2460 | 0.6602 | 49.21% | 84.57% | 24 | 44 |
| stegastamp/full_watermark=True | 85 | 0.4561 | 0.4625 | 91.23% | 91.55% | 0 | 0 |
| stegastamp/full_watermark=False | 85 | 0.0359 | 0.8579 | 7.18% | 77.58% | 24 | 44 |

## Rozkłady — min / P10 / P25 / mediana / P75 / P90 / max

### all

- watermark_fraction_128: 0.0033 / 0.0131 / 0.0446 / 0.4731 / 0.9376 / 0.9822 / 0.9971
- watermark_fraction_512: 0.0031 / 0.0123 / 0.0439 / 0.4711 / 0.9373 / 0.9821 / 0.9969
- bbox_fill: 0.1635 / 0.5497 / 0.7733 / 0.9405 / 0.9942 / 1.0000 / 1.0000
- miou_gt_bbox: 0.2080 / 0.4255 / 0.4688 / 0.4984 / 0.9897 / 1.0000 / 1.0000
### pixelseal

- watermark_fraction_128: 0.0033 / 0.0098 / 0.0506 / 0.4775 / 0.9400 / 0.9800 / 0.9971
- watermark_fraction_512: 0.0031 / 0.0093 / 0.0488 / 0.4761 / 0.9392 / 0.9799 / 0.9969
- bbox_fill: 0.3021 / 0.5928 / 0.7612 / 0.9415 / 0.9938 / 1.0000 / 1.0000
- miou_gt_bbox: 0.2594 / 0.4203 / 0.4712 / 0.5458 / 0.9773 / 1.0000 / 1.0000
### trustmark

- watermark_fraction_128: 0.0038 / 0.0171 / 0.0446 / 0.4711 / 0.9435 / 0.9869 / 0.9965
- watermark_fraction_512: 0.0036 / 0.0168 / 0.0439 / 0.4696 / 0.9426 / 0.9867 / 0.9964
- bbox_fill: 0.2342 / 0.5264 / 0.7850 / 0.9480 / 0.9959 / 1.0000 / 1.0000
- miou_gt_bbox: 0.2080 / 0.4278 / 0.4725 / 0.4981 / 0.9906 / 1.0000 / 1.0000
### stegastamp

- watermark_fraction_128: 0.0038 / 0.0154 / 0.0377 / 0.5020 / 0.9353 / 0.9814 / 0.9968
- watermark_fraction_512: 0.0032 / 0.0150 / 0.0361 / 0.5000 / 0.9348 / 0.9811 / 0.9969
- bbox_fill: 0.1635 / 0.5424 / 0.7802 / 0.9371 / 0.9939 / 1.0000 / 1.0000
- miou_gt_bbox: 0.2825 / 0.4257 / 0.4661 / 0.4976 / 0.9895 / 1.0000 / 1.0000

## Definicje i odtwarzanie

- Pokrycie: liczba jedynek / liczba wszystkich pikseli lub komórek. Raportujemy osobno rozdzielczość 512 i siatkę oceny 128.
- Wypełnienie bbox: liczba jedynek GT / powierzchnia bbox. Nie jest równe mIoU: mIoU uwzględnia też klasę czystą.
- full_watermark=true: zgodnie z loaderem cały obraz był oznakowany, a część przywrócono z czystego oryginału. To nie jest etykieta konkretnego typu edycji.
- Rozkłady obejmują pełne grupy, bez usuwania trudnych przykładów; brak wag po rozmiarze obszaru.
- per_image.csv: wynik każdego obrazu i współrzędne bbox; summary.json: dokładne statystyki, liczby progowe i hashe źródeł.
- Uruchom z task-1: `.venv/bin/python scripts/analyze_geometry.py` (Windows: `.venv\Scripts\python.exe scripts/analyze_geometry.py`).


## Interpretacja

TrustMark: bbox GT daje średnio 0.6621; 28/170 masek odwzorowuje idealnie, 44/170 osiąga co najmniej 0,95. Nie daje więc wyniku bliskiego ideałowi w całej grupie.
Każda metoda ma po 85 przykładów full_watermark=true i 85 false. Rozkład pokrycia jest dwumodalny: małe oznakowane obszary i niemal całe obrazy z czystymi obszarami. Sama średnia około 50% ukrywa ten podział.
Wysokie wypełnienie bbox nie oznacza wysokiego mIoU. Przykład: 90% obrazu ma watermark, a bbox obejmuje cały obraz. IoU watermarka = 0,9, IoU klasy czystej = 0, więc mIoU = 0,45.
Wniosek: bbox dodatniego obszaru warto sprawdzić jako bazę dla części przykładów, ale trzeba osobno modelować czyste obszary i nieregularne kształty. Geometria GT nie mówi jeszcze, czy model potrafi te obszary znaleźć z obrazu. Wyniki train nie gwarantują takich samych wyników test.
