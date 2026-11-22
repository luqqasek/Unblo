# Verified pretrained-forensics candidates

Verified 2026-11-22. Research and source inspection only: no weights downloaded, GPU inference, training, or submission performed by this investigation. Current public baseline supplied by the active experiment owner: **#3005, 0.6597**. Nothing below establishes a route to 0.82.

**Recommendation:** first screen **full IML-ViT adaptation across the three families**, keeping native image pixels; use **SAFIRE source partitions** as the next complementary proposal experiment. Do not repeat frozen TruFor/Mesorch-map fusion under a new name.

## What was already measured here

The canonical checkout contains an extensive completed experiment at `/private/tmp/region-data-results/task-1/experiments/pretrained_forensics/REPORT.md`, plus `PROVENANCE.md`, `source_pins.json` and inference wrappers. This directory is missing from the active main checkout. It evaluated 1,530 frozen maps: three full pretrained localization systems on all 510 real images. It was **not only a Noiseprint++ experiment**. No full forensic-backbone adaptation was performed.

| Frozen model | Direct/inverse macro AUC, all 510 | Direct/inverse clean IoU, watermark >50%, n=255 | Per-image GT-oracle orientation + threshold mIoU, all 510 |
|---|---:|---:|---:|
| TruFor | 0.5706 / 0.4294 | 0.0956 / 0.0170 | 0.5050 |
| Mesorch | 0.5470 / 0.4530 | 0.0929 / 0.0240 | 0.5155 |
| DetectiveSAM | 0.4870 / 0.5130 | 0.0695 / 0.0930 | 0.5109 |

The historical baseline clean IoU on that >50% subset was 0.1380. Direct TruFor/Mesorch hole recall reached 0.9195/0.9895, but low clean IoU and nearly constant outputs prevent interpreting this as successful hole localization. The report does **not** tabulate direct/inverse AUC specifically for the >50% subset; the AUC column above is global and must not be relabeled hole AUC.

Nested frozen-map fusion also failed: old baseline mIoU 0.50624, E1 0.46816, E2 0.47946. E2 versus its own matched control gained 0.00352 [0.00067, 0.00674], but versus original baseline lost 0.02678 [−0.03559, −0.01855]. These are historical pipelines, not matched comparisons against #3005. The genuinely new experiment is **adapting a full pretrained manipulation representation to our actual labels across families**, or using untested source-partition proposals.

## 1. IML-ViT: quickest executable full-model candidate

Official source pin: [`SunnyHaze/IML-ViT@07dd2be`](https://github.com/SunnyHaze/IML-ViT/tree/07dd2be0f4ea27a5c97c9fa5ffbe236733833eac).

**License:** repository MIT; the owner explicitly confirms MIT and commercial use for the official checkpoint in [issue #60](https://github.com/SunnyHaze/IML-ViT/issues/60). This is stronger evidence than inferring weight licensing from the code license.

**Artifact:** [`iml-vit_checkpoint.pth`](https://drive.google.com/file/d/1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7/view), Drive ID `1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7`. SHA-256 reported in that issue: `7631fe852e139d904ad96fd766a37120d703c966cb80dfad70ffae4ff426cce4`. Verify downloaded bytes against it; this investigation has not independently hashed the checkpoint. The [official checkpoint folder](https://drive.google.com/drive/folders/1Ztyiy2cKJVmyusYMUlwuyPecBefTJCPT) also contains training-protocol variants; do not silently substitute them.

**Exact integration:** the [official demo](https://github.com/SunnyHaze/IML-ViT/blob/07dd2be0f4ea27a5c97c9fa5ffbe236733833eac/Demo.ipynb) constructs `iml_vit_model()` with default `input_size=1024`, and strictly loads a bare state dict. Its fallback accepts a training checkpoint's `['model']`; inspect keys rather than swallowing mismatches. The [padding transform](https://github.com/SunnyHaze/IML-ViT/blob/07dd2be0f4ea27a5c97c9fa5ffbe236733833eac/utils/iml_transforms.py#L231) preserves native pixels: right/bottom zero-pad RGB uint8 to 1024, then ImageNet normalization. Do not pad with normalized zero, which differs from normalizing black padding. No input resize is necessary.

The [model forward](https://github.com/SunnyHaze/IML-ViT/blob/07dd2be0f4ea27a5c97c9fa5ffbe236733833eac/iml_vit_model.py#L122) currently accepts `(image, mask, edge_mask)` and returns `(loss, sigmoid_probability, edge_loss)`. For inference, dummy zero labels do not affect prediction. For training, bypass its padded-region BCE and use the project loss on the valid top-left 512 area. Crop output to 512 before averaging 4×4 cells into 128. Output is manipulation probability, **not guaranteed watermark probability**. Freeze any direction/fusion calibration using training data; never choose direct/inverse per validation image from GT.

**Dependencies/feasibility:** core forward imports PyTorch, `timm.models.layers`, and `fvcore`; no Detectron2 installation appears necessary from inspected imports. The original requirements pin albumentations 1.3.0; a manual torch padding/normalization wrapper avoids its version drift. Validate timm `Mlp`/`DropPath` compatibility and strict checkpoint keys. Start with a measured microbatch: 1024 padding and global attention cost more than native512; no throughput claim is verified. Do not change `input_size=512` casually: positional parameter dimensions differ. Upstream forward contains debug prints, which should be removed in an integration wrapper.

## 2. SAFIRE: novel source partitions, with polarity pitfalls

Official source pin: [`mjkwon2021/SAFIRE@ec2af6a`](https://github.com/mjkwon2021/SAFIRE/tree/ec2af6af8c789fb6359493a3e352f89c6fd8107f). The [author's license update](https://github.com/mjkwon2021/SAFIRE#️-license) dated 2026-08-05 explicitly assigns **Apache 2.0 code / CC-BY 4.0 weights**, permitting commercial use with attribution. Original dataset and third-party rights remain separate; no new dataset download is needed for our screening.

The official [Drive folder](https://drive.google.com/drive/folders/1NRxep2G42OnVwCR9sGdf1iPqhCUrGmv2) exposes shortcuts. Resolved target metadata:

| Artifact | Actual target ID | Metadata size |
|---|---|---:|
| [safire.pth](https://drive.google.com/file/d/1KVbMC7z6eG7xDMoSbvelqD7JW3R0koSg/view) | `1KVbMC7z6eG7xDMoSbvelqD7JW3R0koSg` | 420,798,714 bytes |
| [safire_encoder_pretrained.pth](https://drive.google.com/file/d/1NpFWCMLXHnEhgTnp0VeRO2HFiLjKYzZs/view) | `1NpFWCMLXHnEhgTnp0VeRO2HFiLjKYzZs` | 1,076,270,506 bytes |

These are metadata, not downloaded/checksummed files. Use the complete `safire.pth` for localization. [Official inference](https://github.com/mjkwon2021/SAFIRE/blob/ec2af6af8c789fb6359493a3e352f89c6fd8107f/infer_binary.py) uses the repository's modified `segment_anything`, registry `vit_b_adaptor`, plus SAM `sam_vit_b_01ec64.pth`; loads `checkpoint['model']`, stripping `module.`. Ordinary pip SAM is not a substitute. The official [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) is an initialization dependency; verify all final SAFIRE keys load.

**Native512 trap:** default [dataset code](https://github.com/mjkwon2021/SAFIRE/blob/ec2af6af8c789fb6359493a3e352f89c6fd8107f/forgery_data_core.py#L192) resizes to1024. Its `resize_when_large` option instead pads smaller images, preserving uint8 pixels. The predictor also hardcodes 1024 prompts. Pad512→1024 before prediction, restrict prompts and statistics to the valid512 area, and crop masks back. Padding otherwise becomes a spurious dominant source, especially with two-cluster inference.

**Polarity trap:** [binary inference](https://github.com/mjkwon2021/SAFIRE/blob/ec2af6af8c789fb6359493a3e352f89c6fd8107f/networks/safire_predictor_binary.py#L230) clusters masks, chooses each cluster's most confident mask, and treats the smaller selected region as foreground. Its combined foreground map is not a calibrated watermark score. Use **source-mask proposals** and score each with existing watermark evidence. Per-image best polarity or best SAM/source mask chosen by GT is an oracle geometry diagnostic, not deployable performance.

Minimal inference still needs torch/torchvision, numpy, OpenCV, local k-means/SAM, and imports from `safire_model.py` include monai/matplotlib/tqdm. The published environment contains many notebook/medical packages; don't assume all are necessary. SAFIRE is less turnkey than IML-ViT but asks a different question from failed dense frozen-map fusion: can source regions supply missing hole geometry?

## 3. Mesorch and full TruFor: lower priority / excluded

[Mesorch](https://github.com/scu-zjz/Mesorch) already has a working project wrapper and measured failure above. It offers `mesorch-98.pth` and `mesorch_p-118.pth` through its [official folder](https://drive.google.com/drive/folders/1jwYv-S3HAZqzz0YxM9bJynBiPv-O9-6x), with MIT code, Python3.10, imdlbenco and numpy<2. Separate weight-license confirmation was not found in this check; preserve its third-party backbone notices. It is a fallback for full adaptation, not a novel zero-shot candidate. Its standard versus Permute-F1 scripts must not be conflated: GT-selected polarity does not transfer to our two-class mIoU.

Full [TruFor's license](https://github.com/grip-unina/TruFor/blob/main/TruFor_train_test/LICENSE.txt) restricts use to informational/nonprofit purposes. Do not add it to a cash-prize deployment without appropriate permission. Its old experimental results remain useful historical evidence; code availability alone does not authorize new competition deployment.

## Concrete next screen

1. Keep the existing fixed validation fold and current per-image baseline predictions. Audit cross-family source siblings before combining training partitions.
2. Strict-load IML-ViT, verify raw512/padded1024 tensors and output alignment on two images. Run a small frozen-map sanity check, including direct/inverse diagnostics, without fitting validation polarity.
3. Train the **full pretrained model** on only the corresponding fold-training images across all families, with native pixels and per-image both-class objective. Use one bounded training run, record optimizer exposure and actual throughput. Prediction target is our provided mask, not the upstream generic manipulation label. Existing synthetic replay can regularize; avoid real validation masks or a 510-mask bank leaking into training.
4. Compare paired mIoU, per-family deltas, clean IoU and complete misses for >50% and ≤5% clean regions, plus small-watermark regressions. An apparent hole-recall gain from predicting nearly all clean is rejected. Trainable fusion with #3005 predictions must use training-only predictions/calibration.
5. If adaptation supplies no useful hole improvement, inspect SAFIRE proposal oracle coverage first; advance only if source geometry is complementary, then learn deployable polarity/selection using training images. Cache proposals and reuse them. Never report source-partition oracle or Permute-F1 as leaderboard-capable mIoU.

No reported paper benchmark is used as an expected competition score. Both candidates remain unmeasured on our task; the proposed advantage is a testable hypothesis, and the archived negative frozen-map results lower expectations for a large zero-shot jump.

## SAFIRE native512 source-proposal implementation review

Follow-up, 2026-11-22: the experiment owner reports IML's 150-update screen failed (blend mIoU **0.64138 versus 0.65992**; no submission). The following is inspected SAFIRE code, not measured proposal quality or throughput. It supersedes the earlier suggestion that a separate SAM initializer is required.

**Full checkpoint is sufficient if strict loading succeeds.** [Training saves the complete wrapper state dict](https://github.com/mjkwon2021/SAFIRE/blob/ec2af6af8c789fb6359493a3e352f89c6fd8107f/train.py#L345), including frozen encoder/prompt parameters; [inference loads `checkpoint['model']` strictly](https://github.com/mjkwon2021/SAFIRE/blob/ec2af6af8c789fb6359493a3e352f89c6fd8107f/infer_multi.py#L55). Construct `sam_model_registry['vit_b_adaptor'](checkpoint='')`: **empty string, not `None`**. The [builder](https://github.com/mjkwon2021/SAFIRE/blob/ec2af6af8c789fb6359493a3e352f89c6fd8107f/segment_anything/build_sam.py#L283) explicitly accepts `''`, whereas `None` reaches `Path(None)`. Then strip only leading `module.` and strict-load the full state. Download neither the encoder-only file nor the SAM initializer. Artifact completeness remains a runtime assertion, not permission to use `strict=False`.

For minimal inference, the registry's returned `Sam` can itself receive that state: [Sam](https://github.com/mjkwon2021/SAFIRE/blob/ec2af6af8c789fb6359493a3e352f89c6fd8107f/segment_anything/modeling/sam.py#L44) and [AdaptorSAM](https://github.com/mjkwon2021/SAFIRE/blob/ec2af6af8c789fb6359493a3e352f89c6fd8107f/networks/safire_model.py#L39) register the same three submodules and nonpersistent normalization buffers. Their preprocess constants are identical. With input/original sizes both1024, Sam's additional crop/resize is identity geometry. This avoids importing monai/matplotlib/tqdm solely through `safire_model.py`; confirm a small numerical parity check if the wrapper is changed.

Minimal extraction outline, using **the pinned repository's** `segment_anything`:

```python
from segment_anything import sam_model_registry
from segment_anything.predictor_safire import SamPredictor
from segment_anything.utils.amg import build_point_grid

model = sam_model_registry['vit_b_adaptor'](checkpoint='')
checkpoint = torch.load('safire.pth', map_location='cpu', weights_only=True)
state = {k.removeprefix('module.'): v for k, v in checkpoint['model'].items()}
model.load_state_dict(state, strict=True)
model = model.eval().cuda()
predictor = SamPredictor(model)
# image_rgb: original HWC uint8, exactly512x512; no normalization beforehand.
padded = np.zeros((1024, 1024, 3), np.uint8)
padded[:512, :512] = image_rgb
x = torch.from_numpy(padded).permute(2, 0, 1)[None].cuda().float()
with torch.inference_mode():
    predictor.set_torch_image(x, (1024, 1024))  # skips ResizeLongestSide
    points = torch.as_tensor(build_point_grid(16) * 512, device='cuda').float()
    for batch in points.split(32):
        logits, quality, _ = predictor.predict_torch(
            batch[:, None, :], torch.ones((len(batch), 1), device='cuda', dtype=torch.int),
            multimask_output=False, return_logits=True)
        logits = logits[:, 0, :512, :512]  # crop before statistics/storage
        # Stream valid-region binary masks/probabilities to CPU; release full outputs.
```

This preserves all original pixels and samples 256 positive prompts at 32-pixel spacing entirely inside valid512; no GT-derived prompts. Run FP32 first, including the high-frequency encoder, and profile. Avoid calling the original dataset loader, which normally resizes512→1024. A standard `SafirePredictor.generate` alternative accepts `points_per_side=None, point_grids=[build_point_grid(16)*0.5]`, but retains full1024 logits and probabilities for every prompt: approximately **2 GiB just for those two float32 arrays per image**, before masks/features. Streaming is preferable.

**Keep all sources; do not call binary foreground selection.** Every point-prompt mask is a candidate source region. The [multi-source path](https://github.com/mjkwon2021/SAFIRE/blob/ec2af6af8c789fb6359493a3e352f89c6fd8107f/networks/safire_predictor_multi.py#L229) groups prompt masks by average encoder feature, chooses highest predicted-quality mask per cluster, and softmaxes representative logits across clusters. This keeps all clusters; the binary path's smaller-region rule must not be used. For clustering, intersect each mask with valid512 **before** downsampling to the encoder's64 grid and computing its feature mean; skip empty feature supports, retain their raw candidates, and seed NumPy (upstream k-means randomly initializes). Treat clusters as unlabeled source IDs. Never identify “clean” using GT in deployment. First oracle screen can use the raw proposals without clustering, eliminating that extra failure mode.

**Screen specification:** generate the same fixed prompt bank for every fixed-fold image; only afterward use GT to report clean-hole diagnostics. Report best single proposal IoU against `1-mask128`, recall at0.5/0.7, and oracle gain over baseline, separately for clean≤1%, clean1–5%, and watermark>50%. Evaluate a predeclared candidate set containing each proposal and its valid-region complement (a surrounding-source mask may delineate a hole). Match the grid exactly: native binary proposal `logit>0` →4×4 majority; for a complement, downsample the complemented512 mask separately because 8/16 ties make `down(~m)` differ from `~down(m)`. Also report continuous probability-averaged128 diagnostics separately. Do not create arbitrary GT-chosen unions of proposals. This is an **oracle geometry ceiling**, not deployable mIoU; any successful next step needs GT-free region selection/polarity.

Minimal direct-predictor dependencies inspected: torch, torchvision, NumPy, Pillow, local SAM modules. Adding the official multi-source wrapper imports OpenCV, scikit-learn and local k-means. No Detectron2 runtime or monai is needed for the direct path. No local clone, weight download, or GPU execution was performed during this follow-up.
