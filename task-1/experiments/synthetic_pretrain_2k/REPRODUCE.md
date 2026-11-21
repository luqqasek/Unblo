# Odtworzenie eksperymentu 2k

Uruchamiać z katalogu głównego repozytorium. Istniejące wyniki są wznawiane; zmiana konfiguracji lub źródeł blokuje mieszanie starych i nowych checkpointów. Dla innej konfiguracji użyć osobnego katalogu eksperymentu.

```powershell
task-1/.venv/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/download.py
task-1/.venv/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/prepare.py
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/run.py --device cuda:1
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/retention.py
task-1/.venv/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/report.py
task-1/.venv/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/verify.py
task-1/.venv/Scripts/python.exe task-1/dashboard/build.py
```

Można osobno uruchomić `run.py --variant control` lub `--variant synthetic`. GPU 1 to GTX 1080 Ti użyta w obu ramionach, PyTorch 2.5.1+cu121, float32, TF32 wyłączone, operacje deterministyczne. Generacja i ekstrakcja: CPU, ONNX Runtime, 4 wątki. `retention.py` używa tego samego GPU 1.

`results/environment_cpu.txt` i `results/environment_gpu.txt` zawierają wersje pakietów. `config.json` zapisuje hashe kodu treningowego, bazy cech i istniejących checkpointów; `generation_config.json` kod generatora i encoder; `sources/downloads.json` archiwa COCO i ich SHA-256. Początkowy nieudany preflight (zbyt ciasna tolerancja float32 1e-6) zachowano w `generation_preflight_initial_config.json`. Maksymalna różnica cech wobec bazy wyniosła 2.63e-6; kontrola przy atol/rtol 1e-5 przeszła. W tamtym momencie nie wygenerowano żadnych próbek.

Najważniejsze pliki:

- `manifest.json`: 2200 źródeł, wiadomości, maski, statystyki jakości, transformacje STN; train=2000, validation=200.
- `images/`, `masks/`: bezstratne dane 512×512. `shards/`: 22 paczki po 100 przykładów z cechami float32 i oficjalnymi maskami 128×128.
- `pretrain_seed*.pt`: 10-epokowy pretraining, bez selekcji na real170.
- `control_fold*_seed*`, `synthetic_fold*_seed*`: checkpoint, historia selekcji wewnętrznej i predykcje OOF; po dziewięć przebiegów na wariant.
- `synthetic_only_seed*.npz`: transfer bez fine-tuningu, stały próg 0.5.
- `cache/*synthetic_holdout.npz`: kontrola zachowania lokalizacji syntetycznej po treningu na real.
- `per_image_metrics.csv`, `paired_metrics.csv`, `summary_metrics.csv`, `size_stratified_metrics.csv`: rzeczywista ocena OOF.
- `synthetic_holdout_metrics.csv`: syntetyczna walidacja z jawnymi NaN dla AP/AUC/Top-K pełnych i pustych masek.
- `summary.json`, `verification.json`, `REPORT.md`: wynik, kontrole i interpretacja.

Kod treningu nie czyta konkursowego splitu test. Istniejące walidacyjne checkpointy są tylko hashowane i odczytywane, a aktywne podziały kopiowane bez zmian. Dashboard kopiuje tylko raporty, CSV/JSON i wybrane wykresy/galerię, bez archiwów COCO, pełnego syntetycznego zbioru i modeli.
