# Task 1 · Watermark localization

**Final result: public mIoU 0.7309, accepted submission 4129.** [Receipt](solution/upload_receipt.json) · [exact accepted masks](solution/submission-4129.pt) · [challenge brief](challenge.pdf).

Given a 512×512 image, watermark family and known message, predict a binary 128×128 map. The score averages watermarked and clean IoU. Small watermark regions and small clean holes both matter.

## Final solution and code

| Component | Selected method | Relevant code |
| --- | --- | --- |
| Shared image evidence | Two-member DINO recipe: rank-32 and family-conditioned 2,000-step models; separate partial-mark and clean-hole heads | [DINO model/training](experiments/h100_hour/dino_job4.py), [fusion](experiments/h100_hour/fuse_eval.py) |
| PixelSeal specialist | EMA588 initialization; feature-CutMix decoder at step 200, output gain 8; FP32 identity-view inference | [Decoder training](experiments/manual_weights/decoder_refine.py), [portable model](solution/model.py) |
| Final masks | PixelSeal 70% frozen ensemble + 30% specialist; expected-mIoU decoding; frozen TrustMark/StegaStamp masks | [Decoder](experiments/h100_hour/h2_decode.py), [replay/inference](solution/reproduce.py) |
| Artifact contract | Complete expected test names, binary uint8 masks and immutable file hash | [Validation](verify_submission.py), [format helper](src/watermark_localization/submission.py) |

The final solution supersedes the earlier 0.6777 submission. TrustMark/StegaStamp use the frozen two-member DINO recipe, which differs from the four-member ensemble in earlier submission 3707. The public gain therefore cannot be attributed solely to the final PixelSeal weight edit.

## Verify and replay

```bash
# From repository root. Python 3.11 recommended.
python3 -m venv task-1/.venv
task-1/.venv/bin/pip install -r task-1/solution/requirements.txt
task-1/.venv/bin/python task-1/verify_submission.py

# Download final weights and replay maps from this submission repository.
gh release download final-4129 --repo luqqasek/Unblo \
  --pattern pixelseal-featuremix-4129.pt --pattern replay-inputs-4129.npz \
  --dir task-1/solution/assets

task-1/.venv/bin/python task-1/solution/reproduce.py replay --out /tmp/unblo-replayed.pt
```

Replay recomputes the blend and decoding and requires every mask and image name to match the accepted submission. Serialization bytes may differ while tensors remain identical. The canonical accepted file is **16,734,635 bytes**, SHA-256 `58a10ea69eb3a179530795208a0ad69da9a8350592e894ff799e6789fe5ee48b`.

To rerun the actual PixelSeal checkpoint on official test images:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 task-1/.venv/bin/python task-1/solution/reproduce.py infer \
  --data /path/to/official-data --device mps --out /tmp/unblo-inferred.pt
```

The dataset directory must contain `test/metadata.json` and `test/images/`. CPU and CUDA device options are available. The portable loader needs no JURECA account, original resume checkpoint or watermark embedder download.

**Reproduction scope:** the complete replay covers the official 1,020 test IDs. PixelSeal neural output can be recomputed, while its ensemble inputs and the other families use saved test maps. This is not a fresh-image inference service for all three families. Historical training source is included, but its cached features and intermediate training checkpoints are not distributed as a complete retraining pipeline.

## Validation

| Evidence | Result | Scope |
| --- | --- | --- |
| Organizer response | **0.7309 mIoU**, previous best 0.7205 | Public split, submission 4129 |
| PixelSeal final edit | 0.629340 → **0.630891**, delta +0.001552 | Same 57 repeatedly reused development images |
| Paired delta interval | **[−0.003158, +0.007130]** | Includes zero; original promotion gate **failed** |
| Portable loader | Strict state load; maximum probability error **4.77e−7** on three images | Historical comparison against original model wrapper |
| Exact artifact | 1,020 names; uint8 [1020,128,128]; binary masks; hash matches receipt | Checked again during packaging |

The local PixelSeal screen and public three-family result answer different questions. No statistically reliable local gain or 0.80 result was established. The final hidden score remains unknown. [Development evidence](solution/checkpoint-result.json) · [loader checks](solution/portable-loader-verification.json) · [test inference record](solution/inference.json) · [artifact comparison](solution/preupload-validation.json).

## Provenance

Imported from [MelidaBraier/watermark-localization-4129](https://github.com/MelidaBraier/watermark-localization-4129), revision `ad3e936ffd9b54b916b6d21c533ae7abed0595c6`. The five vendored VideoSeal modules retain their [MIT license](solution/vendor/videoseal/LICENSE), upstream commit `b1bf4c081e40bc8971f072f33e3b16912a35f3b4`. [Manifest](solution/manifest.json) records the final model and replay hashes.
