# Reproduce the archived validation

This is a partial, zero-training PCMV diagnostic. It is not a trained PCMV model,
a full Stage 0 pass, or a prediction of public mIoU. Read REPORT.md for the exact
scope and separate the GT-assisted ranking AUC from mask mIoU.

From the repository root, using a Python environment with NumPy:

```sh
python -m unittest discover -s task-1/experiments/pcmv_validation/tests -v
python task-1/experiments/pcmv_validation/publish_dashboard.py
```

These commands validate the archived evidence and rebuild both dashboard languages.
They do not train, run CUDA inference, download weights, or overwrite measured JSON.

The measured inference scripts are `proposal_screen.py` and `signal_oracle.py`.
They require the official train images/masks, the original PixelSeal model and
local experiment caches/weights on the RTX machine. `proposal_screen.py` requires
SAM2 and the exact PS baseline NPZ; see its recorded configuration and hashes in
`results/proposal_screen.json`. `signal_oracle.py` additionally requires clean
synthetic TRAIN covers and their records from `rtx_all_information`.
`make_gallery.py` and `summarize_masks.py` require the baseline NPZ and cached
proposal masks. Those large local caches, trained weights and official datasets
are not part of this source publication. Running these scripts recomputes their
output files; keep the archived measurements separate when changing settings.
Proposal caches are keyed by image name: use a fresh `results/proposals` directory
when changing generator settings or the baseline checkpoint.

`validate_cached.py` uses the archived `next_ideas/results/oracle/*.json` and SciPy
to reproduce earlier diagnostic arithmetic; it writes three diagnostic JSON files.

The adjacent `rtx_all_information` directory contains the source of the stopped
matched experiment. Its `run.py`, `train.py`, `prepare.py` and `prepare_all.py` are
historical training/data-preparation entry points, not part of validation or the
dashboard build. They are not executed by the commands above. StegaStamp training
was never completed. Publication does not resume any experiment.
