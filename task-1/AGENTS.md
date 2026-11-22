# Task 1 experiment dashboard

For model optimization, start with `experiments/RESEARCH_HANDOFF_2026-11-22.md`
and `experiments/README.md`. Current best accepted submission is #3485 (0.6777
public mIoU): PixelSeal EMA1561 global/carrier plus fixed empty-mask local
fallback (#3457); unchanged TrustMark; StegaStamp450→1000 adaptation/refit.
PixelSeal single-fold validation0.585528; StegaStamp matched57-image score
0.762273→0.784104, NOT full OOF. Tiny PixelSeal remains largely unresolved.
See `carrier_fallback/REPORT.md`, `ss_continuation/REPORT.md` and
`data_expansion/REPORT.md` under experiments. Preserve the
historical inverse-STN correction only in the legacy feature pipeline; the new
raw-image segmentator has no STN. Read the protocol before comparing scores.

After completing a Task 1 experiment, evaluation, or leaderboard submission,
update its report and the dashboard before reporting completion to the user.
Follow `dashboard/README.md`: add an adapter in `dashboard/build.py` when the
result schema is new, rebuild the dashboard, and verify the generated data
includes the new metrics and any accepted submission receipt. Keep holdout,
cross-validation, and public leaderboard scores under their respective protocols.

User preference: maintain the dashboard in Polish by default, with an English
version available. After adding experiments or updating leaderboard results,
rebuild both language versions and verify that the new results appear. This is
a standing requirement for future sessions in this repository.
