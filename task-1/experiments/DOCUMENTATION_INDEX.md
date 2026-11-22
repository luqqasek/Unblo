# Task 1 documentation index — 2026-11-22

This index consolidates the repository history and previously local research reports.
It is a snapshot of saved evidence, not a fresh leaderboard query or a new experiment.
The latest recorded best accepted submission is **#3054, public mIoU 0.6677**.
See the [submission report](next_ideas/SUBMISSION_450_REPORT.md) and
[experiment overview](README.md) for its recipe and validation limitations.

## Start here

- [Research handoff](RESEARCH_HANDOFF_2026-11-22.md): architecture, completed attempts, constraints and remaining problems.
- [Setup and task instructions](../README.md), [setup verification](../SETUP_STATUS.md), and [repository audit](../AUDIT.md).
- [Experiment dashboard](../dashboard/README.md): build and browse the saved reports in Polish or English.
- [Geometry analysis](../analysis/geometry/report.md): ground-truth diagnostics; these are not test-time predictions.
- [Model-specific small-region research](../analysis/model_specific_small_regions_research.md): compatibility of published approaches with the unchanged competition watermark encoders.

## Submission progression

The values below describe whole-submission public scores. Do not compare them directly
with per-family holdout, reused-fold OOF, oracle, or synthetic-dev measurements.

| Submission | Public mIoU | Evidence |
|---|---:|---|
| #1496 | 0.5235 | [Synthetic pretraining submission](synthetic_pretrain_2k/SUBMISSION_REPORT.md) |
| #2163 | 0.6145 | [Message localizer](message_localizer/REPORT.md) |
| #2409 | 0.6369 | [Small-region corrections](small_zoom/REPORT.md) |
| #2675 | 0.6440 | [Local-presence submission](local_presence/SUBMISSION_REPORT.md) |
| #3005 | 0.6597 | [StegaStamp adaptation](next_ideas/REPORT.md) |
| #3054 | 0.6677 | [450-update submission](next_ideas/SUBMISSION_450_REPORT.md) |

#3054 retains PixelSeal and TrustMark from #2675 and replaces StegaStamp with
450-update real/replay adaptation plus the validated empty/full-mask fallback.
These development folds have been reused; the reports do not establish an independent
estimate of hidden-final performance.

## Additional work now archived

| Work | Recorded outcome and scope | Documentation |
|---|---|---|
| Best-model error audit | 397 validation images under three different protocols; identifies small positive regions and clean holes. Historical #2675 snapshot. | [Audit](best_model_error_audit/REPORT.md), [method research](best_model_error_audit/METHOD_RESEARCH.md) |
| Full bit evidence | Frozen-feature RTX diagnostic; no confirmed gain, and historical checkpoint replay remains a limitation. | [Report](rtx_bit_evidence/REPORT.md) |
| PixelSeal proposals | Short RTX NRGMark adaptation; real baseline mIoU remains 0.55748. No accepted verified corrections or submission. | [Plan](rtx_pixelseal_proposals/PLAN.md), [report](rtx_pixelseal_proposals/REPORT.md) |
| All-information model | Completed PS/TM three-fold comparisons: deltas -0.003989 / -0.003753, both intervals include zero. SS stopped during preparation. | [Protocol](rtx_all_information/PROTOCOL.md), [report](rtx_all_information/REPORT.md) |
| PCMV preliminary validation | No selector training. Restricted proposal oracle gain +0.107918 on 57 PS images misses the proposed +0.15 gate. GT-based signal diagnostics motivate further clean-hole checks, not a deployable gain. | [Report](pcmv_validation/REPORT.md), [research](pcmv_validation/RESEARCH.md) |
| PCMV follow-up planning | Source review and fixed-fold protocol; a plan is not proof that the proposed model was trained. | [Sources](pcmv/SOURCES.md), [protocol](pcmv/PROTOCOL.md) |
| IML-ViT clean-hole adaptation | Stopped by the predefined no-gain gate at update 150: mean mIoU 0.659924 → 0.641382 on 171 fixed-fold images. No test inference or submission. | [Protocol](iml_holes/PROTOCOL.md), [completed report](iml_holes/REPORT.md) |
| SAFIRE follow-up | Inspected native-resolution proposal implementation; no GPU execution or measured proposal quality in this review. | [Source review](next_ideas/VERIFIED_FORENSICS_RESEARCH.md) |
| Post-#2409 research brief | Historical bottleneck analysis, preserved with its original evaluation scope. | [Brief](small_zoom/RESEARCH_BRIEF_AFTER_2409.md) |

## Evidence and reproduction boundaries

This documentation update archives reports and selected JSON/CSV/PNG evidence alongside
existing repository history. Model weights, prediction arrays, training caches, datasets,
virtual environments and generated dashboard output remain outside this commit.
Some reproduction commands in archived reports reference experiment scripts held in the
original working checkout or on the RTX/cluster machine; this documentation snapshot
alone is not a complete executable reproduction package for those experiments.
Historical validation, runtime and numerical-check claims are recorded results, not tests
rerun during documentation publication. Literature notes retain their original source links
and review dates; publication of these notes does not imply a new literature review.

The two other challenge workspaces have their own recorded status:
[Task 2 — Model Equality](../../task-2/README.md) and
[Task 3 — Unlearning](../../task-3/README.md).

## Complete experiment document inventory

The following inventory includes reports, protocols, research notes, reproduction guides,
and handoffs available in this snapshot. Each document retains its own date and scope.

- **baseline**: [README](baseline/README.md)

- **best_model_error_audit**: [METHOD_RESEARCH](best_model_error_audit/METHOD_RESEARCH.md), [REPORT](best_model_error_audit/REPORT.md)

- **cluster_cleanup**: [REPORT](cluster_cleanup/REPORT.md)

- **decoder_evidence**: [REPORT](decoder_evidence/REPORT.md)

- **diagnostic_audit**: [FREQUENCY_REPORT](diagnostic_audit/FREQUENCY_REPORT.md), [README](diagnostic_audit/README.md), [REPORT](diagnostic_audit/REPORT.md), [architecture_audit](diagnostic_audit/architecture_audit.md), [final_research_report](diagnostic_audit/final_research_report.md), [preprocessing_audit](diagnostic_audit/preprocessing_audit.md), [stn_alignment_audit](diagnostic_audit/stn_alignment_audit.md)

- **frozen_extractor**: [CV3_REPORT](frozen_extractor/CV3_REPORT.md), [REPORT](frozen_extractor/REPORT.md)

- **head_ablation**: [REPORT](head_ablation/REPORT.md), [RESEARCH](head_ablation/RESEARCH.md)

- **iml_holes**: [PROTOCOL](iml_holes/PROTOCOL.md), [README](iml_holes/README.md), [REPORT](iml_holes/REPORT.md)

- **known_message_audit**: [README](known_message_audit/README.md), [REPORT](known_message_audit/REPORT.md), [RESEARCH](known_message_audit/RESEARCH.md)

- **local_message**: [REPORT](local_message/REPORT.md), [RESEARCH](local_message/RESEARCH.md)

- **local_presence**: [PLAN](local_presence/PLAN.md), [REPORT](local_presence/REPORT.md), [SUBMISSION_REPORT](local_presence/SUBMISSION_REPORT.md)

- **loss_screen**: [REPORT](loss_screen/REPORT.md)

- **message_localizer**: [PLAN](message_localizer/PLAN.md), [REPORT](message_localizer/REPORT.md)

- **native_hybrid_submission**: [REPORT](native_hybrid_submission/REPORT.md)

- **native_localizer**: [README](native_localizer/README.md), [REPORT](native_localizer/REPORT.md), [STATUS](native_localizer/STATUS.md)

- **next_ideas**: [MESSAGE_EXTENSION_REPORT](next_ideas/MESSAGE_EXTENSION_REPORT.md), [PROTOCOL](next_ideas/PROTOCOL.md), [REPORT](next_ideas/REPORT.md), [SUBMISSION_450_REPORT](next_ideas/SUBMISSION_450_REPORT.md), [SUBMISSION_REPORT](next_ideas/SUBMISSION_REPORT.md), [VERIFIED_FORENSICS_RESEARCH](next_ideas/VERIFIED_FORENSICS_RESEARCH.md)

- **official_localization**: [README](official_localization/README.md), [RESEARCH](official_localization/RESEARCH.md)

- **official_localization_cv**: [README](official_localization_cv/README.md)

- **pcmv**: [PROTOCOL](pcmv/PROTOCOL.md), [SOURCES](pcmv/SOURCES.md)

- **pcmv_validation**: [REPORT](pcmv_validation/REPORT.md), [RESEARCH](pcmv_validation/RESEARCH.md)

- **pretrained_forensics**: [PROVENANCE](pretrained_forensics/PROVENANCE.md), [REPORT](pretrained_forensics/REPORT.md), [REPRODUCE](pretrained_forensics/REPRODUCE.md)

- **regime_moe**: [CONCLUSIONS](regime_moe/CONCLUSIONS.md), [PHASE_A](regime_moe/PHASE_A.md), [PHASE_B](regime_moe/PHASE_B.md), [README](regime_moe/README.md), [REPORT](regime_moe/REPORT.md), [SCREENING](regime_moe/SCREENING.md)

- **region_data**: [PLAN](region_data/PLAN.md), [REPORT](region_data/REPORT.md)

- **remark_localization**: [README](remark_localization/README.md), [REPORT](remark_localization/REPORT.md), [RESEARCH](remark_localization/RESEARCH.md)

- **research_handoff**: [REPORT](research_handoff/REPORT.md)

- **rgb_srm**: [CONCLUSIONS](rgb_srm/CONCLUSIONS.md), [README](rgb_srm/README.md), [REPORT](rgb_srm/REPORT.md)

- **rtx_all_information**: [PROTOCOL](rtx_all_information/PROTOCOL.md), [REPORT](rtx_all_information/REPORT.md)

- **rtx_bit_evidence**: [REPORT](rtx_bit_evidence/REPORT.md)

- **rtx_pixelseal_proposals**: [PLAN](rtx_pixelseal_proposals/PLAN.md), [REPORT](rtx_pixelseal_proposals/REPORT.md)

- **small_region_diagnosis**: [HANDOFF_A100](small_region_diagnosis/HANDOFF_A100.md), [README](small_region_diagnosis/README.md), [REPORT](small_region_diagnosis/REPORT.md)

- **small_zoom**: [PLAN](small_zoom/PLAN.md), [REPORT](small_zoom/REPORT.md), [RESEARCH_BRIEF_AFTER_2409](small_zoom/RESEARCH_BRIEF_AFTER_2409.md)

- **stegastamp_diagnosis**: [ANALYSIS_REPORT](stegastamp_diagnosis/ANALYSIS_REPORT.md), [REPORT](stegastamp_diagnosis/REPORT.md)

- **stegastamp_stable_selection**: [REPORT](stegastamp_stable_selection/REPORT.md)

- **stegastamp_stn_alignment**: [REPORT](stegastamp_stn_alignment/REPORT.md)

- **synthetic_pretrain_2k**: [README](synthetic_pretrain_2k/README.md), [REPORT](synthetic_pretrain_2k/REPORT.md), [REPRODUCE](synthetic_pretrain_2k/REPRODUCE.md), [RESEARCH](synthetic_pretrain_2k/RESEARCH.md), [SUBMISSION_REPORT](synthetic_pretrain_2k/SUBMISSION_REPORT.md)

- **synthetic_scaling**: [README](synthetic_scaling/README.md), [REPORT](synthetic_scaling/REPORT.md), [REPRODUCE](synthetic_scaling/REPRODUCE.md)

- **triforensic_wm**: [PROTOCOL](triforensic_wm/PROTOCOL.md), [REPORT](triforensic_wm/REPORT.md)

- **trustmark_patches**: [README](trustmark_patches/README.md)
