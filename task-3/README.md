# Task 3 · Machine unlearning

**Selected result: E50_feature10, public score 0.3149, accepted submission 4183.** [Server receipt](evidence/submissions-E50_feature10_queue_v1_attempt1.json) · [GPU validation](evidence/validation-E50_feature10.json) · [challenge brief](challenge.pdf).

Starting from the supplied ViT-B/16 checkpoint, forget 3,500 unlabeled images while retaining utility. The evaluator compares exact and altered forget images with retrained gold models and applies a 70% hidden-accuracy gate. Gold models, control images and utility labels are unavailable locally.

## Selected method and code

**Forget by a controlled amount for each image.** Rank image hardness using the original model's confidence under fixed altered views. Convert that rank to target log-odds with mean −3 and spread 6.5. Preserve the teacher's relative distribution over the other classes. Altered forget views receive an additional one-logit reduction.

Fine-tune against those targets while preserving teacher outputs on strong transformed/mixed views (weight 5) and normalized CLS features (weight 10). The selected E50 configuration uses only the supplied checkpoint and forget images; external natural anchors and synthetic-bank replacement belong to other candidates.

| Component | Code |
| --- | --- |
| Official model and pixel preprocessing | [common.py](src/common.py) |
| Hardness ranking and target construction | [targets.py: hardness_v2 / target_v2](src/targets.py) |
| Exact, mild and preservation views | [aug.py](src/aug.py) |
| Distillation and CLS-feature preservation | [unlearn_core.py: run_unlearn](src/unlearn_core.py) |
| Recorded configuration and diagnostics | [E50 run record](evidence/runs-E50_feature10.json) |
| One-command selected training recipe | [train_best.sh](scripts/train_best.sh) |
| Challenge formula with genuine labels/gold logits | [metrics.py](metrics.py) |

## Reproduction

```bash
cd task-3
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_assets.py
bash scripts/train_best.sh
# On CUDA, check 256 real images under server-style FP16 inference:
python scripts/validate_gpu.py outputs/runs/E50_feature10.pt --report outputs/validation.json
python -m unittest discover -s tests -v
```

The wrapper computes original logits and 28 probe views, then runs 2,500 updates with batch 32, learning rate 6e-5, AdamW and cosine decay. It exports a float32 state dict in the original architecture and does not upload. The historical training used PyTorch 2.5.1+cu124, timm 1.0.31 and an A100. GPU numerical variation means a retrained checkpoint need not match the historical hash bit for bit.

## Validation

- Accepted public score **0.3149**, checkpoint SHA-256 `74f11cb4d86cc8a7ceae42dce6949fb750a14b5e20bdfbeaddb5038d8115c35f`.
- Historical GPU checks: strict architecture load, finite float32 tensors, **343,629,322 bytes**, finite FP16 output on **256 real images**.
- Local diagnostic exact pseudo-label CE **4.2717**, altered CE **3.8203**. These use original-model predictions as labels and are not ground-truth accuracy.
- Seven metric tests cover score composition, per-gold statistics, the utility gate, undefined denominators and invalid inputs.

E50 increases normalized CLS-feature preservation weight from E30's 0.5 to 10 while retaining its other selected settings. Public results: E30 **0.3013**, E43 (weight 2) **0.3055**, E50 (weight 10) **0.3149**. The [16-candidate comparison](evidence/round12-comparison.csv) preserves the local evidence. These candidates were adaptively developed; public improvements do not establish hidden-test generalization.

The exact selected checkpoint is available in the [Unblo release](https://github.com/luqqasek/Unblo/releases/tag/final-4129), outside Git's source tree:

```bash
gh release download final-4129 --repo luqqasek/Unblo --pattern E50_feature10.pt --dir outputs/final
python scripts/verify_submission.py outputs/final/E50_feature10.pt
```

Its local hash matches both the receipt and GPU validation record. The final hidden score and true retain accuracy are unknown. No new competition submission is sent by repository checks.
