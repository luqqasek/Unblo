# Shared IML-ViT clean-hole experiment

[Predeclared protocol](PROTOCOL.md), [verified primary-source research](../next_ideas/VERIFIED_FORENSICS_RESEARCH.md).
Based on the official [IML-ViT](https://github.com/SunnyHaze/IML-ViT), MIT code and checkpoint ([owner confirmation](https://github.com/SunnyHaze/IML-ViT/issues/60)). Upstream source and its LICENSE remain alongside downloaded weights on the cluster; weights are not committed here.

Dependencies added to the existing pinned task environment: `gdown==6.5.2`, `fvcore==0.1.5.post20221221`. Existing torch/timm versions retained. Source/weight provenance is saved by setup; runtime versions saved by training.

```bash
PYTHONPATH=.:src .venv-native/bin/python -m experiments.iml_holes.setup --root /p/project1/training2666/pszenny1/native-localizer
PYTHONPATH=.:src .venv-native/bin/python -m experiments.native_localizer.submit_cluster --development --minutes 30 --gpus 4 -- experiments.iml_holes.train --root /p/project1/training2666/pszenny1/native-localizer
```

Run from cluster `task-1`. Submit helper refuses another queued/running job for this user. GPU execution occurs only under Slurm. This screen depends on existing split manifests, exact encoded PNG training pools, historical label-disjoint baseline maps and newly computed GT-independent full-image decoder scores for every validation image; it does not create a new training corpus or read test labels.

No result has been measured at project creation. Report/dashboard must be updated from actual saved outputs before declaring success.
