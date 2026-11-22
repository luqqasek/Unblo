# Unblo

European AI Championship Paris 2026. Three tasks about what models reveal, what they share, and what they can forget.

## Results and code

| Task | Best confirmed public result | Our solution | Start here |
| --- | --- | --- | --- |
| 1. Watermark localization | **0.7309 mIoU**, submission **4129** | DINO ensemble with a PixelSeal feature-CutMix specialist and expected-mIoU decoding | [Method, code and validation](task-1/README.md) |
| 2. Model equality | **1.0000**, submission **2411** | Separate behavioral disguises from weight changes using response statistics and local likelihood evidence | [Method, code and replay](task-2/README.md) |
| 3. Machine unlearning | **0.3149**, submission **4183** | Per-image forgetting targets with output and CLS-feature preservation | [Method, training and validation](task-3/README.md) |

These are saved **public-split** results, not final hidden-test scores. Each task links to its accepted server receipt. Local development scores use different protocols and are reported separately.

## Presentation

[Download the presentation PDF](slides/Unblo.pdf) or [open the offline HTML deck](slides/index.html) after cloning. **Nine slides, exactly three per task.** Use arrow keys to present the HTML version. [Speaker notes](slides/NOTES.md) include the technical qualifications and code pointers.

## Verify the submission

No competition API key or new endpoint query is required for these checks.

```bash
git clone https://github.com/luqqasek/Unblo.git
cd Unblo

# Task 1: install its environment, then verify the exact accepted masks.
python3 -m venv task-1/.venv
task-1/.venv/bin/pip install -e './task-1[dev]'
task-1/.venv/bin/python task-1/verify_submission.py
task-1/.venv/bin/python -m pytest task-1/tests -q

# Task 2: locked lightweight environment and reconstruction of all 33 labels.
uv sync --project task-2 --frozen
task-2/.venv/bin/python task-2/replay.py
task-2/.venv/bin/python -m unittest discover -s task-2/tests -p 'test_*.py'

# Task 3: scoring-formula tests require only NumPy.
task-1/.venv/bin/python -m unittest discover -s task-3/tests -v
```

[Validation record](VALIDATION.md) distinguishes checks rerun for this submission from historical GPU and competition results. GPU training instructions and artifact requirements live in each task README.

## Repository map

```text
task-1/    Selected localization source, accepted masks, receipt, validation
task-2/    Imported model-equality solution, frozen evidence, accepted labels
task-3/    Selected E50 unlearning code, exact configuration, receipt, validation
slides/    Presentation PDF, offline HTML source and speaker notes
```

Datasets, model downloads, environments and intermediate tensors are excluded. The accepted Task 1 prediction file and Task 2 CSV are included. Task 1's specialist/replay assets and Task 3's exact selected 344 MB checkpoint are attached to the [submission release](https://github.com/luqqasek/Unblo/releases/tag/final-4129), with hashes in the task evidence.

[Research decisions](RESEARCH_NOTES.md) summarize rejected approaches. [Provenance](PROVENANCE.md) identifies imported work and upstream sources. The full pre-cleanup research history remains in Git, rather than competing with the final solution in the working tree.

## Submission details

**Team:** Unblo

**Repository:** https://github.com/luqqasek/Unblo

Text for the organizers' `general` channel:

> Unblo — https://github.com/luqqasek/Unblo

The repository must be public at the organizers' noon deadline. Posting to the channel is a separate submission step.
