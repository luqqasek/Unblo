# Task 2 · Model equality

**Selected result: public score 1.0000, accepted submission 2411.** [Server receipt](submission/receipt.json) · [accepted 33-endpoint CSV](submission/labels.csv) · [challenge brief](challenge.pdf).

Identify endpoints that share a base model family and endpoints that share the same checkpoint, quantization and LoRA adapter. System prompts, decoding changes and watermarks are disguises of the same model. The task weights family MCC by 0.1 and within-family model MCC by 0.9.

## Selected method

1. Assign model families from response evidence, including manual analysis.
2. Detect behavioral disguises: custom prompt evidence, missing maker names, repeated openings, unusually varied vocabulary and digit-distribution changes.
3. Compare saved responses under local base, adapter and emulated quantized models. A sufficiently better likelihood indicates changed weights.
4. Apply the fixed v4 decision order. Adapter evidence takes priority, followed by quantization evidence, disguise grouping, and temperature/plain-base heuristics.

| Step | Relevant code |
| --- | --- |
| Behavioral features and thresholds | [typed2.py](research/experiments/s82/typed2.py) |
| Adapter likelihood aggregation | [v3.py: lora_gains](research/experiments/s82/v3.py) |
| Final grouping rule | [v4.py: label](research/experiments/s82/v4.py) |
| Original manual tags and family assembly | [scored_v4.py](research/experiments/s82/scored_v4.py), [families.json](research/experiments/s82/families.json) |
| Local likelihood measurement | [wb_score.py](research/experiments/s82/wb_score.py), [quant_llr.py](research/experiments/s82/quant_llr.py) |
| Exact historical replay | [replay.py](replay.py) |
| Submission format and scoring | [scoring.py](src/meq/scoring.py) |

## Offline reproduction

```bash
# From repository root:
uv sync --project task-2 --frozen
task-2/.venv/bin/python task-2/replay.py
task-2/.venv/bin/python -m unittest discover -s task-2/tests -p 'test_*.py'
# Local fake-server integration test, no competition requests:
MEQ_API_KEY=test-key task-2/.venv/bin/python task-2/tests/test_offline.py
```

Replay uses the saved likelihood measurements and original tags, reconstructs all 33 labels, checks every pair's grouping against the accepted CSV, and verifies its SHA-1 against the upload receipt. It neither queries endpoints nor uploads. Later Qwen-3B measurements in the upstream repository were not part of this historical submission and are excluded from replay.

## Validation and limits

The [submission history](evidence/submissions.jsonl) records public results 0.55, 0.55, 0.55, 0.30 and **1.00**. The imported [prediction evidence](evidence/predictions.json) records practice **1.0000** and test-bed **0.487638**, with known families, combining to **0.679774** under 37.5%/62.5% weights. These are development results, not hidden-score estimates. [Heuristic explanation](evidence/heuristics.html).

**Human decisions are part of the solution:** the submitted family/tag assignments were not fully automatic, and `ep-26` was excluded from plain-base attachment after public leaderboard feedback. The perfect public score does not prove every hidden pair is correct. Reconstruction establishes the final labels, not a complete historical trace of every intermediate decision.

The raw response captures are included to preserve irreplaceable evidence. The original endpoint allowance was 100 responses per scored endpoint. No new competition queries were made for this packaging work. New likelihood computation additionally requires local base/adapter weights and Torch/Transformers/PEFT; the locked environment is sufficient for replay, rules and tests.

Imported from [MelidaBraier/model_equality](https://github.com/MelidaBraier/model_equality); see [import provenance](evidence/import.json).
