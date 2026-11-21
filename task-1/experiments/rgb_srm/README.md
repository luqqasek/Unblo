# RGB / SRM: kontrolowane porównanie reprezentacji

Plan zapisany przed screeningiem: A (zamrożony ekstraktor), B (RGB), C (RGB+SRM),
D (ekstraktor+RGB+SRM). Osobne modele dla każdej metody. Cztery lekkie bloki
16/24/32/32, rozdzielczości 256/128/64/32. Trzy stałe filtry high-pass osobno
na każdym kanale RGB obrazu 512; wejście CNN 256. Bez dopasowania normalizacji
do danych, augmentacji ani danych syntetycznych. Filtry to mały bank SRM-style,
nie pełny Spatial Rich Model.

W B/C istniejący Head dostaje cztery mapy CNN, z identycznym dekoderem i wyjściem
128. W D cztery projekcje RGB/SRM są dodawane do projekcji zamrożonych cech
przy 32×32; warstwy fuse istniejącego Head pozostają identyczne.

Wszystkie warianty: float32 CUDA, TF32 off, batch 8, AdamW 0.001 / decay 0.0001,
0.5 BCE + 0.5 Dice, maks. 50 epok, patience 8. Ta sama selekcja i zapisane
wewnętrzne/zewnętrzne podziały: PS/TM 80/20; SS trzy wewnętrzne foldy.
StegaStamp A/D korzystają z uprzednio zweryfikowanego inverse-STN cache.

Najpierw fold1, seed42 dla A/B/C. B/C przechodzą do pełnego CV danej metody,
jeśli jednocześnie poprawiają AP i Top-K przy GT ≤5%, a macro AP nie spada o
więcej niż 0.01 względem A. Kryterium nie używa globalnego mIoU.
A/D zawsze: 3 foldy × seedy 42/7/123. Raport osobno pokazuje fold screeningowy
i pozostałe foldy. To rozwojowe porównanie istniejącego zbioru, nie nowy test.

Uruchomienie z katalogu repozytorium:

```powershell
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/rgb_srm/run.py
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/rgb_srm/report.py
task-1/.venv/Scripts/python.exe task-1/dashboard/build.py
```

Wyniki w `results/`: konfiguracja i hashe źródeł, splity, screening, historie,
checkpointy, probabilities float32, maski, per-image CSV, tabela porównań,
podsumowanie JSON i galerie. Dotychczasowe najlepsze artefakty są tylko czytane;
ich hashe sprawdzane po treningu. Bez predykcji testowych i bez submission.
