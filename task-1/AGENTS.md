# Task 1 experiment dashboard

For model optimization, start with `experiments/RESEARCH_HANDOFF_2026-11-22.md`
and `experiments/README.md`. Current best accepted submission is #3054 (0.6677
public mIoU): PS/TM retained from #2675, including NRGMark crop verification and
the empty-mask TM scan; native RGB StegaStamp after450 real/replay updates and
the validated fixed empty/full-mask fallback. See `next_ideas/REPORT.md`. Preserve the
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
