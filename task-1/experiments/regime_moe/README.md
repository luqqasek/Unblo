# Regime-aware MoE — reprodukcja

Eksperyment korzysta z istniejących 510 realnych obrazów, cache ekstraktorów,
aktywnych checkpointów inverse-STN StegaStamp i dotychczasowych foldów.
Nie wykorzystuje istniejących syntetyków ani danych testowych.

Środowisko: istniejące `task-1/.venv-rgb-cuda`, PyTorch 2.5.1+cu121,
torchvision 0.20.1+cu121; zależności analityczne ze środowiska projektu.
GPU0: RTX3070, GPU1: GTX1080Ti. Domyślnie poniższe skrypty używają GPU1.
Ustawienie `--device 0` wybiera RTX. Trening używa float32, TF32 off.

Z katalogu repozytorium:

```powershell
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/regime_moe/phase_a.py
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/regime_moe/phase_b.py
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/regime_moe/phase_c.py --stage screen
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/regime_moe/phase_c.py --stage confirm
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/regime_moe/report.py
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/regime_moe/verify.py
task-1/.venv/Scripts/python.exe task-1/dashboard/build.py
```

Faza A zapisuje wszystkie porównania klasyfikatorów oraz wybrany model każdego
outer foldu. W fazie B najpierw powstają inner held-out mapy bazowe. Faza C
wymaga raportów A/B. Potwierdzenie rozszerza tylko rodziny z pozytywnym screeningiem.
Seedy 42/7/123; progi reżimów i współczynnik wag 4 są stałe. Parametry i reguły
awansu zapisane w `results/config.json` przed pierwszym treningiem gate.

Fazy B/C obsługują `--methods pixelseal trustmark` lub `--methods stegastamp`.
Można uruchomić rozłączne rodziny na różnych GPU. Nie uruchamiać dwóch workerów
na tym samym foldzie/seedzie: współdzielą nazwy artefaktów.
Ukończone treningi są wznawiane z wyników; nie używać starego results po zmianie
algorytmu. Do nowej konfiguracji należy utworzyć osobny katalog eksperymentu.

`A0`: zamrożony uniwersalny model, `A1`: kalibracja reżimowa, `A2`: S+M,
`A3`: M+L, `A4`: S+M+L soft, `A5`: uniform, `A6`: GT oracle (diagnostyka),
`Hard`: wybór przez argmax gate, `U0`: dalszy trening bez specjalizacji.
`S_only` i `L_only` pokazują samodzielne działanie specjalistów.

Artefakty `results/`: OOF probabilities/maski/progi, mapy ekspertów i gate,
joblib klasyfikatorów, checkpointy specjalistów (odwołanie do zamrożonego M),
inner manifesty i mapy, historie selekcji, per-image/fold/seed CSV, parowane CI,
wykresy, galeria oraz weryfikacja. Wszystkie GT-derived pola są treningiem lub
diagnostyką; wdrożeniowe A1–A5/Hard nie pobierają area/reżimu GT jako wejścia.
