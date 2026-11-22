# Submission validation

Checked on 22 November 2026. These checks use saved artifacts and local fixtures; they do not submit to the competition or query scored endpoints.

| Check | Result | What it establishes |
| --- | --- | --- |
| Task 1 accepted artifact | PASS, 1,020 binary uint8 masks, SHA-256 matches submission 4129 | Exact submitted file and complete expected image coverage |
| Task 1 cached replay | PASS, all 1,020 masks identical | Final blend and decoding reproduce the accepted tensors |
| Task 1 release assets | PASS, both SHA-256 hashes match the upstream manifest | Checkpoint and replay-map integrity |
| Task 1 portable checkpoint | PASS, strict load and finite CPU output on one real test image | Loader works in the packaged directory layout |
| Task 1 tests | **33 passed** | Metrics, official compatibility, pixel geometry, split checks, source separation, blending and orchestration contracts |
| Task 2 frozen replay | PASS, **33/33** labels and all pair partitions match | Historical v4 result reproduced without later Qwen-3B evidence |
| Task 2 unit tests | **21 passed** | Statistical evidence, permutation behavior and development-score safeguards |
| Task 2 fake-server integration | PASS, nine scenarios | Retry handling, budget caps, resumability, prompt-ID conflicts and submission/scoring checks |
| Task 2 local benchmark | Practice **1.0000**, test bed **0.487638**, weighted **0.679774** | Reproduced development metrics with known families |
| Task 3 selected checkpoint | SHA-256 matches E50's accepted receipt and historical GPU validation | Local release file identifies accepted submission 4183 |
| Task 3 tests | **9 passed** | Scoring formula, utility gate, finite inputs, protected-head behavior and normalized CLS-feature gradients |
| Presentation | Nine slides, three per task; rendered PDF and browser previews | Layout, readable text and no detected text overflow |

Task 1's one-image CPU output differs from the cached MPS probability map by at most **0.00012073**. This is a smoke check, not a full backend-equivalence claim. The exact cached replay is the certified path for reproducing all accepted masks. See [packaging record](task-1/solution/packaging-check.json).

Historical Task 3 GPU validation establishes strict architecture loading, finite FP32 weights and finite FP16 logits on 256 images. The packaging work does not re-estimate hidden retain accuracy or rerun the training. Public results come from the preserved organizer receipts, not from the unit tests.

## Commands

```bash
task-1/.venv/bin/python task-1/verify_submission.py
task-1/.venv/bin/python task-1/solution/reproduce.py replay --out /tmp/unblo-replayed.pt
PYTHONPATH=task-1:task-1/src task-1/.venv/bin/python -m pytest task-1/tests task-1/experiments/h100_sprint/tests -q
task-2/.venv/bin/python task-2/replay.py
task-2/.venv/bin/python -m unittest discover -s task-2/tests -p 'test_*.py'
MEQ_API_KEY=test-key task-2/.venv/bin/python task-2/tests/test_offline.py
task-2/.venv/bin/python task-2/research/experiments/s82/bench4.py
task-1/.venv/bin/python -m unittest discover -s task-3/tests -v
node slides/render.cjs
```

The fake server listens only on localhost. Rendering requires Google Chrome and the Playwright Node package. The recorded packaging environment used Python 3.11/PyTorch 2.5.1/timm 1.0.31 for Tasks 1 and 3, and the locked Task 2 packages under Python 3.13. Task 1's supplied upstream inference requirements pin timm 1.0.12; both the original pins and this packaging-environment difference are preserved explicitly.
