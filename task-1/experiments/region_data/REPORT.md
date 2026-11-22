# Dane obiektowe i małe pozostałości — dopasowane porównanie

Baza: message_localizer. Architektura, checkpoint startowy, budżet i loss identyczne; zmieniona receptura danych. Nie użyto testu konkursowego ani leaderboardu.

| Metoda | Etap | Dane | mIoU | FG IoU ≤1% | FG IoU 1–5% | FG IoU >5% | Clean IoU dziury ≤1% | Clean IoU dziury 1–5% |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pixelseal | synthetic | control | 0.55500 | 0.00000 | 0.03851 | 0.68634 | 0.00129 | 0.07693 |
| pixelseal | synthetic | objects | 0.56919 | 0.00000 | 0.03666 | 0.73127 | 0.01228 | 0.06685 |
| pixelseal | adapt | control | 0.58566 | 0.00000 | 0.04615 | 0.74318 | 0.01227 | 0.09446 |
| pixelseal | adapt | objects | 0.58548 | 0.00000 | 0.03781 | 0.74624 | 0.01228 | 0.09503 |
| pixelseal | position | objects_plus100 | 0.56841 | 0.00000 | 0.03428 | 0.73054 | 0.00083 | 0.07720 |
| pixelseal | position | position_pairs100 | 0.56253 | 0.00000 | 0.04365 | 0.72566 | 0.00000 | 0.07510 |
| trustmark | synthetic | control | 0.58415 | 0.00000 | 0.06114 | 0.77413 | 0.00681 | 0.15598 |
| trustmark | synthetic | objects | 0.58291 | 0.00000 | 0.07059 | 0.76850 | 0.00746 | 0.14381 |
| trustmark | adapt | control | 0.58761 | 0.00000 | 0.07153 | 0.79130 | 0.00259 | 0.12445 |
| trustmark | adapt | objects | 0.59333 | 0.00000 | 0.07164 | 0.78847 | 0.00568 | 0.14776 |
| trustmark | position | objects_plus100 | 0.58693 | 0.01634 | 0.08201 | 0.78567 | 0.00056 | 0.13674 |
| trustmark | position | position_pairs100 | 0.58288 | 0.00163 | 0.09007 | 0.78338 | 0.00182 | 0.11524 |

## Różnice sparowane

- pixelseal / synthetic_paired: ΔmIoU +0.01419, bootstrap95% [0.00608786842342355, 0.022124720271050915], n=170.
- pixelseal / adapt_paired: ΔmIoU -0.00017, bootstrap95% [-0.003695120662708723, 0.003462050702624213], n=170.
- pixelseal / position_paired: ΔmIoU -0.00589, bootstrap95% [-0.01110286001840656, -0.0007455430502572677], n=170.
- trustmark / synthetic_paired: ΔmIoU -0.00124, bootstrap95% [-0.006550690305304699, 0.0034439027515333637], n=170.
- trustmark / adapt_paired: ΔmIoU +0.00573, bootstrap95% [0.00048249871390260954, 0.010739717790448826], n=170.
- trustmark / position_paired: ΔmIoU -0.00405, bootstrap95% [-0.010562089089557957, 0.0020927883715831513], n=170.

## Protokół i ograniczenia

Dwie kontynuacje po200kroków×32widoki od tego samego checkpointu400. LR5e-6/5e-5. Obiektowa gałąź:50% stare dane,25% same-content z maskami COCO,25% transplantacje/dodania między zdjęciami. Nowe osadzenia128train+32dev/metodę w kwadratach128/192/256/384; brak nowych źródeł, siła i wrapper bez zmian. Bez post-embedding JPEG/blur/resampling.

Adaptacja:150kroków/fold, batch24synthetic+8real, LR2e-6/2e-5,3oryginalne foldy, stały próg0,5. Obie gałęzie mają identyczną sekwencję realnych przykładów dzięki osobnemu RNG. Zapisane nowe dane odpowiadają całej recepturze, nie izolują wpływu samych masek od zmiany sampling weights i skali osadzania.

Drugi test:100 dodatkowych kroków od obiektowego checkpointu600, identyczne LR/loss/batch, RNG73. Porównuje dalszy stary sposób parowania z równopowierzchniowymi parami pozycyjnymi, skoncentrowanymi na małych regionach i dziurach. RGB nie jest przesuwane ani skalowane; zmienia się jedynie support na tym samym dokładnie oznakowanym źródle. Obie maski pary mają identyczną powierzchnię także po oficjalnym downsamplingu. To synthetic-only i adaptacyjny etap rozwojowy.

Raport obejmuje rozwojowe OOF, wielokrotnie używane do badań. Bootstrap jest warunkowy po obrazach i nie uwzględnia ponownego treningu/całej zależności źródeł. Wynik na0 lub1 obrazie w małej grupie nie jest stabilnym dowodem. Nie zmieniono produkcyjnego#2163.

Artefakty JURECA: `/p/project1/training2666/pszenny1/native-localizer/region-runs/`; dokładne dodatkowe osadzenia: `region-data/`. Konfiguracje, realizowane proporcje, provenance i geometria/sygnał w `results/runs/`. Kod: prepare.py, data.py, experiment.py, adapt.py. Wszystkie etykiety pochodzą z pochodzenia pikseli, nigdy z niezerowego residualu.

Reprodukcja w przydziale4GPU: `python -m experiments.region_data.job`, potem `python -m experiments.region_data.adapt_job`. Lokalna analiza po pobraniu JSON: `PYTHONPATH=task-1:task-1/src task-1/.venv/bin/python -m experiments.region_data.analyze`.

## Geometria i sygnał — diagnostyka

| Metoda | Źródło | n partial | Wypełnienie bbox | Komponenty | Mediana szerokości, średnia po obrazach (komórki128) |
|---|---|---:|---:|---:|---:|
| pixelseal | real | 170 | 0.6298 | 2.32 | 8.83 |
| pixelseal | control | 112 | 0.8124 | 9.88 | 5.90 |
| pixelseal | objects | 121 | 0.6386 | 9.26 | 5.91 |
| trustmark | real | 170 | 0.6030 | 2.54 | 9.22 |
| trustmark | control | 113 | 0.7756 | 12.10 | 7.69 |
| trustmark | objects | 122 | 0.6521 | 7.25 | 6.29 |

Próbka syntetyczna:128 primary/dev/gałąź, niezależna od stałych64 przykładów oceny modelu. Komponenty:4-connectivity. Nowe dane przybliżają bbox fill, ale nie usuwają rozbieżności liczby komponentów i szerokości. To opis rozkładów, nie dowód przyczyny wyniku.

| Metoda | Reżim osadzenia | n dev | RMS | Udział niezmienionych kanałów |
|---|---|---:|---:|---:|
| pixelseal | intermediate_crop_remnant | 32 | 1.4395 | 0.3439 |
| pixelseal | large_source_remnant | 95 | 1.3602 | 0.3390 |
| pixelseal | local_crop | 33 | 0.9442 | 0.5664 |
| trustmark | intermediate_crop_remnant | 32 | 1.9021 | 0.3488 |
| trustmark | large_source_remnant | 102 | 2.1556 | 0.3268 |
| trustmark | local_crop | 26 | 1.7433 | 0.3337 |

RMS w jednostkach uint8, bez wzmacniania. Udział zerowych kanałów różni się od udziału całych niezmienionych pikseli; oba zapisano oddzielnie w audit_summary.json, gdzie dostępne. Rzeczywistych clean originals nie mamy, więc nie porównujemy tych residuali bezpośrednio z realnymi. Globalne bit accuracy małych fragmentów jest bliskie0,5 mimo niezerowego residualu; takie przykłady pozostają w treningu.
