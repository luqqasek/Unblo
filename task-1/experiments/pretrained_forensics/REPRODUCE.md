# Reproduction

Run from the repository root. Prerequisites: the 510 real training images,
official `data/train/masks_128.npz`, frozen_extractor/results/cv3 OOF artifacts,
and synthetic_pretrain_2k/results/synthetic_fold{1,2,3}_seed42.npz.
These existing artifacts are read-only. No competition test files are needed.

Python 3.11, torch 2.5.1+cu121, torchvision matching that torch, NumPy, SciPy,
scikit-learn, pandas, matplotlib, Pillow, requests, timm, PyYAML, OpenCV, tqdm.
The run used `task-1/.venv-rgb-cuda/Scripts/python.exe` and RTX3070 CUDA device0.
Exact base/extra package snapshots are in `results/environment_{base,extra}.txt`.
Those are PowerShell redirect outputs (UTF-16 on Windows PowerShell5).

```powershell
$py = 'task-1/.venv-rgb-cuda/Scripts/python.exe'
$ex = 'task-1/experiments/pretrained_forensics'
& $py -m pip install --target "$ex/results/deps" --no-deps -r "$ex/requirements-extra.txt"
& $py "$ex/setup_models.py"
& $py -u "$ex/infer_trufor.py"
& $py -u "$ex/evaluate.py" --model trufor
& $py -u "$ex/infer_mesorch.py"
& $py -u "$ex/evaluate.py" --model mesorch
& $py -u "$ex/infer_detectivesam.py"
& $py -u "$ex/evaluate.py" --model detectivesam
& $py "$ex/report.py"
# DetectiveSAM passed the recorded C/D gate in this run:
& $py -u "$ex/crossfit_baseline.py"
& $py -u "$ex/fusion.py"
& $py "$ex/fusion_report.py"
& $py "$ex/verify.py"
& $py task-1/dashboard/build.py
```

Model setup downloads official weights and pins source commits. See
[provenance](PROVENANCE.md) for author links and license details.
`setup_models.py` never replaces existing files; existing hashes are verified.
Inference scripts skip existing per-image files. If changing preprocessing or
weights, use a new experiment directory, not a stale cache. Models load strictly.
Mesorch only bypasses the training registry import/decorator and uses the actual
MesorchFull class; all numerical source code is unchanged.
SAM2 is imported directly from its pinned source, without its optional CUDA
extension. The official DetectiveSAM runner and preprocessing are used; **never
pass GT masks**, because its noise seed incorporates mask metadata.

Outputs: `results/{trufor,mesorch,detectivesam}/<ID>.npz` contain native512 and
evaluation128 continuous maps, binary masks, complements, timings and memory;
TruFor additionally exposes confidence and global detection. Original images
and official128 GT are copied to `results/inputs/{rgb,gt}` once.
`eval_<model>` contains per-image, per-family/area, per-fold metrics, calibration
IDs, direct/inverse/exact-threshold oracle diagnostics, paired bootstrap and gate.
`results/gallery` contains deterministically selected diagnostic examples.
The result bundle also includes `inputs/baseline_oof.npz`, `inputs/splits.json`,
and an exact copy of the official scorer (`reference_metric.py`). Evaluation and
reporting can use those snapshots if the surrounding repository artifacts are
absent. Regenerating nested baseline training inputs additionally requires the
existing frozen feature caches and StegaStamp pretrain checkpoint; these external
prerequisites are hash-identified in the cross-fit input manifests.

No new learned model may be trained before evaluating C/D and writing `gate.json`.
The feasibility criterion is in evaluate.py; inspect actual metrics as well as its
boolean. A passing gate requires a separate inner-cross-fitted fusion comparison;
ordinary global OOF is not valid as outer-training stacking input because its
underlying model may have seen outer-validation labels. A failed gate skips E.

Source images are never transformed into new training examples. Toroidal shifts
in evaluation move **prediction maps** only, serving as a spatial negative control.
Official DetectiveSAM blur/noise branches are internal inference, not synthesis
of a training dataset. All reported oracle choices explicitly use GT and cannot
be deployed.
