# Task 1 experiment dashboard

For model optimization, use the active development baseline documented in
`experiments/README.md`. The user selected inverse-STN StegaStamp as the starting
point for further optimization despite mixed CV results; preserve this geometry
in subsequent variants and compare against its matched fold/seed checkpoints.

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
