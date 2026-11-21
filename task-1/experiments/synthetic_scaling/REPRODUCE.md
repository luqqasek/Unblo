# Odtworzenie eksperymentu

Uruchamiaj z katalogu głównego repozytorium. Eksperyment korzysta z istniejących danych, zamrożonych ekstraktorów i zapisanych podziałów; nie zmienia wcześniejszych wyników. Pliki wynikowe pozostają w `synthetic_scaling/results/`.

## Wymagane źródła

- Zbiór rzeczywisty: `task-1/data/train/`, 510 obrazów.
- Cechy PixelSeal i TrustMark: `frozen_extractor/results/*_features.pt`; StegaStamp: `stegastamp_stn_alignment/results/aligned_features.pt`.
- Etykiety i identyfikatory OOF: `frozen_extractor/results/cv3/*_oof.npz`.
- Podziały: `frozen_extractor/results/cv3/folds.json` i `stegastamp_stn_alignment/results/splits.json`.
- Poprzednia generacja StegaStamp: `synthetic_pretrain_2k/results/manifest.json`, `shards/`, `sources/`, `cache/features.onnx` oraz checkpointy `pretrain_seed*.pt` do kontroli odtwarzalności.
- Oficjalne encodery w `task-1/checkpoints/` oraz zainstalowany pakiet TrustMark. Konkretne pliki i SHA zapisuje `results/source_snapshot.json`.

Badanie implementacji i źródła metody generacji: [poprzedni research](../synthetic_pretrain_2k/RESEARCH.md). Ten eksperyment bada liczebność przy tym samym generatorze; nie stanowi porównania z metodami SOTA na ich benchmarkach.

## Środowisko

GPU: `task-1/.venv-rgb-cuda/Scripts/python.exe`, PyTorch 2.5.1+cu121. GPU 0: RTX 3070, GPU 1: GTX 1080 Ti. CPU/ONNX i raporty: `task-1/.venv/Scripts/python.exe`. Wersje pozostałych bibliotek zapisano w snapshotach. Cechy mają format float32, są stronicowane przez NPY/memmap; kod przenosi na GPU tylko bieżący batch. Wymagane są dziesiątki GB wolnego dysku na cechy.

## Generacja

```powershell
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/synthetic_scaling/provenance.py
task-1/.venv-rgb-cuda/Scripts/python.exe -u task-1/experiments/synthetic_scaling/generate.py --method pixelseal --device cuda:0
task-1/.venv-rgb-cuda/Scripts/python.exe -u task-1/experiments/synthetic_scaling/generate.py --method trustmark --device cuda:1
task-1/.venv/Scripts/python.exe -u task-1/experiments/synthetic_scaling/generate.py --method stegastamp --device cpu
task-1/.venv/Scripts/python.exe -u task-1/experiments/synthetic_scaling/verify.py --data-only
```

Trzy generatory można uruchomić równolegle w osobnych terminalach. Wznawianie odbywa się po zapisanych blokach 100 obrazów. Konfiguracja musi pozostać identyczna; zmiana generatora wymaga nowego katalogu wyników. Każda metoda ma 4000 przykładów treningowych i 200 osobnej syntetycznej walidacji. Pierwsze 2000 treningowych są wspólną podpulą obu porównań.

## Trening

W osobnych terminalach:

```powershell
task-1/.venv-rgb-cuda/Scripts/python.exe -u task-1/experiments/synthetic_scaling/run_queue.py --gpu 0
task-1/.venv-rgb-cuda/Scripts/python.exe -u task-1/experiments/synthetic_scaling/run_queue.py --gpu 1
```

GPU 0 kolejno wykonuje PixelSeal i TrustMark, GPU 1 StegaStamp. Kolejki czekają na zakończenie potrzebnej generacji. Logi: `results/training_<method>.log`. Nie uruchamiaj dwóch kolejek dla tej samej karty. Można też uruchomić `train.py --method <method> --device cuda:<n>` bez kolejki, zachowując zapisane urządzenie.

Każda metoda: 3 seedy, 2 ciągłe trajektorie pretrainingu na seed, po 2 z góry ustalone checkpointy. Daje to 12 syntetycznych checkpointów i 45 końcowych fitów OOF na metodę. Łącznie 18 trajektorii pretrainingu i 135 końcowych fitów, poza wewnętrznymi treningami wyboru epoki. Stan optymalizatora i generatora losowego jest zapisywany po każdej epoce pretrainingu. Wznowienie pomija ukończone fity OOF. Nie zmieniaj `train.py` ani jego konfiguracji podczas wznowienia.

## Raport i kontrole

Po zakończeniu wszystkich trzech metod:

```powershell
task-1/.venv/Scripts/python.exe -u task-1/experiments/synthetic_scaling/report.py
task-1/.venv/Scripts/python.exe -u task-1/experiments/synthetic_scaling/verify.py
task-1/.venv/Scripts/python.exe task-1/dashboard/build.py
task-1/.venv/Scripts/python.exe task-1/experiments/synthetic_scaling/verify_dashboard.py
```

Ostatnia kontrola zakłada działający lokalny serwer na porcie 8877. Raport: `http://127.0.0.1:8877/reports/synthetic_scaling/REPORT.html`.

Kontrole obejmują wszystkie 12 600 masek syntetycznych, SHA coverów, dokładną zgodność wcześniejszych 2200 przykładów StegaStamp, rozłączność źródeł i foldów, poprawność bbox TrustMark, oficjalną metrykę dla wszystkich predykcji OOF i zgodność raportu z zapisami. Każdy fit dodatkowo odtwarza predykcje z zapisanego checkpointu. Brak zewnętrznego zbioru z nowymi etykietami oznacza, że jest to nadal walidacja rozwojowa.

Nie powstają nowe predykcje testowe ani zgłoszenia do leaderboardu. Wcześniejsze zaakceptowane artefakty są kontrolowane hashem.
