# Diagnostyka małych regionów — Etap 0

[Wyniki i tabele](REPORT.md) · [Proponowana dalsza praca na A100](HANDOFF_A100.md).

Wyłącznie aktualny `message_localizer` PS/TM. Bez treningu, nowych augmentacji
i submission. Artefakty są w ignorowanym `results/`; pełne mapy OOF pozostają
w katalogach bazowego eksperymentu. Raport odróżnia pomiar, oracle i propozycje.

Z głównego katalogu repo, PowerShell:

```powershell
$env:PYTHONPATH = 'task-1'
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m pytest task-1/experiments/small_region_diagnosis/test_diagnostics.py -q
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m experiments.small_region_diagnosis.oof --require-maps
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m experiments.small_region_diagnosis.decoder_probe --family pixelseal
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m experiments.small_region_diagnosis.decoder_probe --family trustmark
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m experiments.small_region_diagnosis.crop_tolerance --family pixelseal
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m experiments.small_region_diagnosis.crop_tolerance --family trustmark
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m experiments.small_region_diagnosis.signal --family pixelseal
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m experiments.small_region_diagnosis.signal --family trustmark
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m experiments.small_region_diagnosis.geometry
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m experiments.small_region_diagnosis.report
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' -m experiments.small_region_diagnosis.verify
& 'task-1/.venv-rgb-cuda/Scripts/python.exe' task-1/dashboard/build.py
```

Wymagane są sześć aktualnych `maps.npz`, zapisane metryki/splity,
konkursowy train, oficjalne lokalne wagi i istniejący `val2017.zip`.
Skrypty nadpisują wyłącznie własne diagnostyczne wyniki, zachowując bazę.

Ocena obejmuje globalne mIoU oraz FG IoU ≤1%, 1–5%, 5–20%, >20% i clean holes.
Clean holes to dodatkowa kohorta z `full_watermark`; dla niej także clean IoU.
Bit accuracy i statystyki Δ nie są nowymi wynikami segmentacji.
