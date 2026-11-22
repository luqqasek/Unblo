# PCMV source and protocol notes

Verified 2026-11-22 from official source; no GPU execution or checkpoint download in this review. Scope: one PixelSeal fixed-fold proposal-level add/carve scorer screen, with matched Control A/B and treatment. SAM2 proposes geometry; it does not establish watermark presence. No expected 0.80 score is supported by these sources.

## SAM2.1 small provenance

- Official repository pin: [`facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4`](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4). This matches the source pin archived in the earlier pretrained-forensics experiment.
- Official [SAM2.1 small checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt). Record SHA-256 after download; none was independently verified here.
- Config argument: `configs/sam2.1/sam2.1_hiera_s.yaml`; [pinned config source](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_s.yaml).
- [Official license statement](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md#license) explicitly covers model checkpoints and code under Apache2.0. Preserve notices, including the separately licensed connected-component code if used.
- [Pinned setup](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/setup.py#L20) requires Python≥3.10, torch≥2.5.1, torchvision≥0.20.1, numpy≥1.24.4, tqdm≥4.66.1, hydra-core≥1.3.2, iopath≥0.1.10, Pillow≥9.4.0. The existing torch2.5.1 / torchvision0.20.1 pair meets these requirements. Keep that pair fixed; install only missing dependencies rather than allowing pip to upgrade the training environment. Optional notebooks/training/video extras are unnecessary.
- [Installation](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/INSTALL.md): `SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps -e /path/to/pinned/sam2` skips optional CUDA connected components. Set hole/sprinkle cleanup explicitly to zero; do not silently differ between machines.

## Exact entrypoints and pixel policy

```python
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

model = build_sam2(
    'configs/sam2.1/sam2.1_hiera_s.yaml', 'sam2.1_hiera_small.pt',
    device='cuda', mode='eval', apply_postprocessing=False)
generator = SAM2AutomaticMaskGenerator(
    model, points_per_side=16, points_per_batch=64,
    crop_n_layers=0, min_mask_region_area=0, output_mode='binary_mask',
    pred_iou_thresh=0.8, stability_score_thresh=0.95,
    use_m2m=False, multimask_output=True)
predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    proposals = generator.generate(original_rgb_uint8.copy())
    predictor.set_image(original_rgb_uint8.copy())
    masks, quality, low_res_logits = predictor.predict(
        point_coords=np.array([[x, y]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        normalize_coords=True, multimask_output=True, return_logits=False)
```

These explicit settings are a bounded screening recipe, not a claim of optimal thresholds. Keep them identical across scorer controls. The [builder](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/build_sam.py#L164) loads `checkpoint['model']` and rejects missing/unexpected keys. `apply_postprocessing=False` disables dynamic single-mask fallback; it is distinct from connected-component cleanup.

The [automatic generator](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/automatic_mask_generator.py#L170) returns original-resolution boolean `segmentation`, predicted quality, stability, coordinates and boxes. Those quality scores estimate segmentation quality, **not probability of a watermark**. The [point predictor](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py#L237) expects original-image `(x,y)` pixels when `normalize_coords=True`; do not normalize coordinates twice. Multiple positive points in one `predict` call jointly prompt one object; they are not independent proposals. For independent seeds, use separate calls or an explicitly verified batched prompt implementation. Seeds may come from GT-free base/carrier maps, never GT masks.

[SAM2 transforms](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/utils/transforms.py#L15) convert RGB0–255, resize to model resolution and apply ImageNet normalization; output masks return to original512 geometry. This resize is permitted **only inside the proposal branch**. Retain a separate unchanged RGB uint8 copy for watermark probes, JND, carrier maps and clean/marked arithmetic. Do not decode SAM's resized/normalized tensor. For isolation/ablation probes preserve the512 canvas and unchanged-region bytes; apply only the declared replacement mask, then the unchanged official decoder wrapper once. Such interventions remain decoder distribution shifts and may create seams despite having no geometric resize.

## Statistical and leakage checks for PCMV

1. **Z-score is a feature, not an established significance level.** Decoder logits/bits, overlapping masks and candidate searches are dependent. Do not assert fair-coin Gaussian false-positive rates or √K SNR gains from them. Estimate wrong-message statistics using the same intervention, crop geometry and score; preserve the correct-message physical-presence target. With N randomizations, an empirical tail estimate `(1+exceedances)/(N+1)` cannot be below `1/(N+1)`.
2. **Calibrate the searched decision.** Taking the maximum over many masks raises chance scores. Null calibration must reproduce candidate selection and maximization, not compare a searched correct-message maximum with a single wrong-message crop. Keep proposal counts, quality gates, null count and feature normalization fixed across Control A/B and treatment. A conservative gate may still miss irregular remnants that retain presence without full payload.
3. **Verification does not determine support.** High bit agreement means some message evidence survives; it does not certify every pixel in a mask/rectangle. An ablation-score drop can reflect context destruction or artificial boundaries. Let supervised action scoring evaluate add/carve benefit; preserve the existing TrustMark verification policy when transferring later.
4. **Fit only on fold-training images.** No validation GT in proposal generation, points, gates, JND cutoffs, feature scaling or scorer thresholds. Historical OOF maps from other folds can themselves come from checkpoints trained on this fixed validation fold; do not silently use those as fold-clean scorer-training features. Use appropriate outer-training-only base/carrier checkpoints for all features, or a documented nested construction. Training on in-sample features instead creates a different calibration problem and must be reported.
5. **Score actions, not only region overlap.** Label each candidate by the actual change in official two-class per-image mIoU after add/carve, including interactions with the baseline mask. Include no-op, duplicates and harmful actions. Balance at the image level so an image with many SAM masks does not dominate. For repeated sequential edits, an independently positive candidate may become harmful after earlier edits; re-evaluate or use a predeclared joint policy.
6. **JND pooling needs uncertainty.** Low-JND absence of carrier is not clean-region evidence. Record sample counts and valid area in each JND stratum, empty strata, and carrier scores separately; fit cutoffs only on training data. Never use known clean twins at inference.
7. **Report gates honestly.** Oracle proposal IoU/coverage is a diagnostic ceiling, not achievable mIoU. Evaluate controls and treatment on identical validation IDs, with per-image paired changes, clean/watermark minority misses and both-class IoU. One repeatedly inspected fold is developmental evidence; bootstrap intervals do not remove model-selection optimism. A larger public-board score is possible, not guaranteed.
