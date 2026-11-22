# Source and artifact provenance

- **Task 1:** final package from [MelidaBraier/watermark-localization-4129](https://github.com/MelidaBraier/watermark-localization-4129), commit `ad3e936ffd9b54b916b6d21c533ae7abed0595c6`, accepted submission 4129/public mIoU 0.7309. Portable paths were adapted to `task-1/solution`; complete expected-name checking was added. Earlier team work remains in this repository through commit `95929b914725323789054d628a72a706e9b0b35c`.
- **Task 2:** imported from [MelidaBraier/model_equality](https://github.com/MelidaBraier/model_equality), commit `d977491` (full hash in `task-2/evidence/import.json`). The final S82/v4 rules and frozen evidence are preserved. `replay.py` reconstructs the historical result without later Qwen-3B evidence. The control center and discarded search rounds are omitted.
- **Task 3:** imported from [MelidaBraier/unlearning](https://github.com/MelidaBraier/unlearning), base commit `e7a3e2c20153cb6143276aad88e6705c95aecd65`, plus the supplied local continuation through E50. E50's run configuration, receipt and validation record identify the selected result. The local continuation was not represented by that upstream commit alone.

The organizers' briefs are included as `task-1/challenge.pdf`, `task-2/challenge.pdf` and `task-3/challenge.pdf`. Official assets come from SprintML's [watermark localization](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task), [model equality](https://huggingface.co/datasets/SprintML/model_equality) and [unlearning](https://huggingface.co/datasets/SprintML/unlearning) datasets.

The slides bundle the Inter typeface (Latin subset from `@fontsource-variable/inter` 5.3.0) under the SIL Open Font License 1.1; the license text is `slides/assets/fonts/OFL.txt`.

Third-party model implementations and weights retain their original licenses. This submission does not grant new rights to those assets. Source hashes for the accepted prediction/checkpoint artifacts are recorded with the receipts.

The cleanup changes navigation, packaging and validation entry points. It does not rerun training or replace the accepted submission files. Local scratch work was moved to a backup outside the repository before removal.
