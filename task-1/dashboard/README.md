# Experiment dashboard

A static, offline-capable journal built from saved Task 1 results. Includes searchable
metric tables, fold details, CSV export, accepted submission scores, reports, and
validation prediction panels. Holdout, cross-validation, and public leaderboard
scores are kept separate because they are not directly comparable.

The opening recommendation selects the highest accepted public submission score
from saved receipts (newest submission wins an exact tie). Its detail view groups
local results by evaluation protocol and links to available prediction galleries,
per-image metrics, reports, and the receipt. These galleries are validation/OOF
evidence, not final test predictions. Missing artifacts are not linked. Curated
problem summaries live in `leader.py`; adding a new winning experiment without
one shows an explicit fallback to its report. The optimization baseline remains
the separate user decision documented in `experiments/README.md`.

## Open locally

From the repository root (Python standard library only):

```sh
python task-1/dashboard/serve.py --port 8877
```

Open **http://127.0.0.1:8877**. The command rebuilds the dashboard before serving it.
Stop with Ctrl+C. On Linux use `python3` if needed.

The **Brief do researchu / Research brief** section links to the Polish handoff:
`http://127.0.0.1:8877/reports/research_handoff/REPORT.html`.
It summarizes completed attempts, unresolved questions, lessons and a research-agent
prompt. Download its Markdown when handing it to an agent without local server access.
Source: `experiments/research_handoff/REPORT.md`; update its dated snapshot when new
evidence changes the conclusions.

## Open through SSH

On the remote machine, in a terminal inside this repository, run:

```sh
python3 task-1/dashboard/serve.py --port 8877
```

For a Windows SSH server use `python` instead of `python3`.
On your own computer, in another terminal, run:

```sh
ssh -N -L 8877:127.0.0.1:8877 USER@SSH_HOST
```

Replace `USER` and `SSH_HOST` with your actual SSH login and hostname. For a
nonstandard SSH port, add `-p PORT` before `USER@SSH_HOST`.
Open **http://127.0.0.1:8877** in your local browser. Keep both terminals running;
Ctrl+C stops each process. If local port 8877 is occupied, use
`-L 8878:127.0.0.1:8877` and open local port 8878 instead.

The server binds only to remote loopback and exposes only `dashboard/dist`, not
the repository, credentials, datasets, or checkpoints. No public HTTP port needs
to be opened.

## Update after an experiment

```sh
python task-1/dashboard/build.py
```

Refresh the browser after rebuilding. There is no automatic polling. The output
is `dist/index.html` with embedded data, plus curated images, reports, and
`data.json`. You can also open `dist/index.html` directly without a server.

Keep new experiments under `task-1/experiments/<name>/`. A `REPORT.md` (preferred)
or `README.md` adds a journal entry automatically; `results/*example*.png` adds
gallery panels. Add a result adapter in `build.py` for a new metrics schema. Only
measured artifacts should populate scores; mark unexecuted comparisons explicitly.

The remote machine must have the experiment result artifacts: ignored local
results do not arrive with a Git clone. Alternatively, build where the results
exist and transfer only the generated `dist` directory. Missing expected artifacts
are listed in the dashboard's Data availability section. Generated output is
ignored by Git; the builder and template are versioned.

## Language and maintenance

The default page (`index.html`) is Polish; `index-en.html` provides English.
Use the Polski / English navigation links to switch. Source reports and labels
inside saved images retain their original language. Update Polish display copy
in `polish.py` when adding new dashboard descriptions.

Report links open generated `.html` reading views with a section index, responsive
tables, related reports, and print styles. Original `.md` files remain available
through the download link. For example, open
`http://127.0.0.1:8877/reports/frozen_extractor/REPORT.html`.
The standard-library renderer in `report_view.py` supports report headings,
tables, lists, fenced code, inline code, emphasis, and links without network access.

The standing instruction in `../AGENTS.md` requires updating the report and
rebuilding both dashboard versions after every experiment, evaluation, or
leaderboard update, and checking that the new evidence appears.

The `diagnostic_audit` adapter includes reused OOF summaries and the matched
raw/aligned StegaStamp probe confirmation. Its report bundle copies CSV/JSON,
figures and the 510-image inspection gallery, excluding the diagnostic cache
and all checkpoint/prediction tensors. Open
`http://127.0.0.1:8877/reports/diagnostic_audit/REPORT.html`.

The `synthetic_pretrain_2k` adapter presents the matched real-only and synthetic→real
StegaStamp results for three folds and three seeds. Its report includes synthetic-only
transfer, source provenance, per-image metrics and the comparison gallery. COCO archives,
full generated datasets, feature caches and checkpoints are excluded from the dashboard.
Open `http://127.0.0.1:8877/reports/synthetic_pretrain_2k/REPORT.html`.

The `synthetic_scaling` adapter reads the completed 2k/4k × 2500/5000-update
comparison for all three watermark families. Its 15 primary rows show real OOF
results after finetuning, averaged over three seeds, with all nine fold/seed
details. Synthetic-only transfer is kept in the report as a secondary diagnostic.
The report bundle includes measured CSV/JSON, figures and example panels, while
excluding generated datasets, feature caches, checkpoints and raw prediction files.
Open `http://127.0.0.1:8877/reports/synthetic_scaling/REPORT.html`.
