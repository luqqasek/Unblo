# Task 1 — Localize Image Watermark

Supervisors: Adam Dziedzic and Franziska Boenisch.
Developers: Simon Hector, Michel Meintz, Louis Kerner, Maitri Shah.

Start coding in `src/watermark_localization/solution.py`.

## Environment

Python 3.11 environment: `.venv`. Activate from this directory:

```sh
source .venv/bin/activate
python -m pytest
python scripts/verify_setup.py
```

Use `.venv/bin/python` as your editor interpreter. To reproduce all downloads and
installation, run `bash scripts/setup.sh`. CPU inference is supported on this Mac;
Juelich uses the original CUDA requirements in `requirements-juelich.txt`.
`requirements-macos.txt` substitutes CPU ONNX Runtime and omits the unavailable
Apple Silicon `decord` wheel (video reading is unnecessary for this image task).

## Where to start

The actual official loader and wrappers are in `wmloc/`. A ready-to-run official
baseline is `submission_template.py`: edit its `predict(item, watermarks=None)`
function to begin experimenting. Run it with:

```sh
python submission_template.py --data data --out outputs/submission.pt
```

The baseline predicts clean everywhere. It measures local train mIoU and saves
all 1,020 test predictions. `SUBMIT` remains `False`; no result is uploaded.
`src/watermark_localization/solution.py` is an optional separate implementation entry point.

```python
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor

train = LocalisationDataset("data", "train")
item = train[0]
wm = load_watermark(item["method"], models_dir="checkpoints", device="cpu")
logits = wm.logits(to_tensor(item["image"]))
features = wm.features(to_tensor(item["image"]))
```

`data/train` and `data/test` contain the extracted images and metadata. Training
masks include full-resolution PNGs and `masks_128.npz`. Weights and the pinned
VideoSeal checkout are in `checkpoints/`. TrustMark Q downloads its encoder,
decoder and remover into `.venv/lib/python3.11/site-packages/trustmark/models/`
on first load. Upstream archives and pristine code
remain in `starter-kit/task-1/`; local baseline and wrappers are tracked in Git.

## Official resources

- Task kit/data: https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task
- General JURECA setup: https://huggingface.co/datasets/SprintML/hackathon
- Local copies: `OFFICIAL_README.txt`, `starter-kit/setup/README.md`.
- Leaderboard from the brief: http://34.147.229.55/leaderboard_page

The shared JURECA setup requires your team names and Juelich accounts. Its scripts
are downloaded for reference; they are intended for JURECA, not for this Mac.

## Task contract

Input images are 512 x 512 PNGs, watermarked using PixelSeal, TrustMark, or
StegaStamp. Predict 1 for watermarked pixels and 0 for clean pixels, including
areas overwritten by edits. Original rectangular watermark subsets have maximum
aspect ratio 16:1 for PixelSeal/StegaStamp and 2:1 for TrustMark; edits can produce
irregular masks. Edits include outpainting, object removal/alteration, addition,
and transplantation.

There are 510 train images (170 per method) with 512/128 ground truth masks,
method labels, and embedded messages. The 1,020 test images (340 per method)
provide method labels and messages, without ground truth.

Evaluation uses a 128 x 128 grid. Each cell represents 4 x 4 pixels and is
watermarked if at least 8 pixels are watermarked. `wmloc.utils` provides the official metric and downsampling functions;
`metrics.py` also implements the brief's rule.
For each image, average clean and watermark IoU, treating an empty union as 1;
then average images equally. `mean_iou` expects masks already on the desired grid.

## Submission

Use `save_submission` from `watermark_localization.submission` with the complete
expected test name list from the official loader. It validates names, duplicates,
binary values, and `[N,128,128]` shape, and stores CPU uint8 masks to fit under
20 MB. Preserve the official image names exactly. Output structure:

```python
{"image_names": names, "masks": masks}
```

Public leaderboard: 30% of test images. Private ranking: remaining 70%.
The official starter kit and validator remain authoritative.

The downloaded official README says 25 MB; the supplied participant brief says 20 MB.
The local submission helper enforces the stricter 20 MB limit.

## Windows

From PowerShell in this directory, with Git and Python 3.11 installed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe submission_template.py --data data --out outputs/submission.pt
```

The installer uses `py -3.11` to create the environment and CPU ONNX Runtime.
Windows execution has not been verified on this Mac. GPU setup is separate.

## Dashboard do przeglądania danych

Dashboard działa lokalnie i nie wymaga dodatkowych bibliotek ponad NumPy i Pillow
z przygotowanego środowiska. Nie ładuje modeli watermarkowania.

Windows, PowerShell w folderze `task-1`:

```powershell
.\.venv\Scripts\python.exe dashboard/server.py
```

macOS / Linux:

```sh
.venv/bin/python dashboard/server.py
```

Otwórz **http://127.0.0.1:8765**. Zakończ serwer przez Ctrl+C.
Możesz zmienić port: `dashboard/server.py --port 8766`.

- Galeria wszystkich obrazów train/test, wyszukiwanie po nazwie i filtrowanie po metodzie.
- Obraz wejściowy, zielona nakładka ground truth i maska binarna obok siebie.
- Przełączanie maski 512 × 512 / oficjalnej siatki 128 × 128, regulacja nakładki.
- Wiadomość bitowa, flaga `full_watermark` i ścieżki plików źródłowych.
- Filtrowanie i sortowanie po pokryciu oraz rozkład pokrycia dla wybranych danych.

Pokrycie liczone jest z oficjalnej maski 128 × 128. Dla testu jest nieznane,
więc dashboard nie pokazuje masek, wartości pokrycia ani flagi `full_watermark`.
Metadane nie zawierają etykiet typu edycji. Podgląd niczego nie zmienia w danych.
