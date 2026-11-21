# ReMark as localization evidence for TrustMark Q

Inspected 2026-11-21. Scope: the installed `trustmark==0.8.0`, Adobe's official repository, and the ICCV 2025 paper. This note records source inspection and verified file hashes; it does not report localization experiment results. No runtime, checkpoints, datasets, or existing scripts were modified.

## Decision

ReMark is a sensible **feature source to test**, not an established watermark localizer. Start with its floating-point correction, and optionally frozen KBNet features. Evaluate against spatial ground truth and clean controls before interpreting any correction magnitude as watermark presence. The installed implementation returns an image and has no mask output or segmentation head. Its architecture also mixes information globally. These conclusions follow from the installed wrapper (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`), remover (`task-1/.venv/Lib/site-packages/trustmark/denoise.py`), and KBNet (`task-1/.venv/Lib/site-packages/trustmark/KBNet/kbnet_s_arch.py`).

## Paper evidence and limits

ReMark is trained as image restoration using the watermark embedder and extractor, with pixel reconstruction, adversarial, and decoder-response losses. Its goal is removal and re-watermarking, not pixel classification. Resolution scaling computes a correction at 256×256 and interpolates that correction onto the original image. Section 4.4/Table 3 reports PSNR 48.48 dB and bit accuracy 0.553 after removal, versus 40.20 dB and 0.970 without removal, for TrustMark with `alpha_max=20` and strength 1.0. These are image fidelity and payload metrics; they do not establish spatial localization accuracy or the performance of the installed Q checkpoint. The paper acknowledges accumulating unwanted noise during repeated re-watermarking. [ICCV paper, sections 3.2–3.3 and 4.4](https://www.collomosse.com/pubs/Bui-ICCV-2025.pdf)

The [CVF publication page](https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html) was discoverable, but direct CVF HTML/PDF retrieval returned HTTP 403. The complete paper was inspected at the coauthor's website, which the official repository also links. No secondary scientific sources were used.

## Version and exact checkpoint

The installed distribution metadata (`task-1/.venv/Lib/site-packages/trustmark-0.8.0.dist-info/METADATA`) identifies version 0.8.0. `TrustMark(model_type='Q')` loads Q-specific encoder, decoder, and remover files. Its constructor always loads the remover. Q is described in source as the quality/robustness variant with a ResNet50 payload decoder; that decoder architecture is distinct from the KBNet remover. Installed source (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`)

| Variant | Remover checkpoint | MD5 declared by installed source |
| --- | --- | --- |
| Q | `trustmark_rm_Q.ckpt` | `760337a5596e665aed2ab5c49aa5284f` |
| B | `trustmark_rm_B.ckpt` | `eb4279e0301973112b021b1440363401` |
| C | `trustmark_rm_C.ckpt` | `5ca3d651d9cde175433cebdf437e412f` |
| P | `trustmark_rm_P.ckpt` | `8b8f4715ea474327921ee9d0f46d2c3f` |

The local Q checkpoint's MD5 was computed with `Get-FileHash` and **matches** the Q declaration. The local `trustmark_rm_Q.yaml` MD5 is `8476bcd4092abf302272868f3b4c2e38`, also matching its declaration. Q and C share the configuration hash, not the weights. No local weights for other variants were loaded or verified. The package's loader checks hashes and may download files if missing or mismatched, so avoid relying on a constructor call as a read-only inspection step. Installed checksum table and loader (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`)

The Q configuration (`task-1/.venv/Lib/site-packages/trustmark/models/trustmark_rm_Q.yaml`) uses `trustmark.denoise.WMRemover` wrapping `trustmark.KBNet.kbnet_s_arch.KBNet_s`: `lightweight=true`, width 32, FFN scale 1.5, 12 middle blocks, encoder block counts `[2,2,4,8]`, and decoder block counts `[2,2,2,2]`. The loader sets `is_train=False`, loads weights with `strict=False`, moves the model to the selected device, and calls `eval()`. Thus a custom loader should inspect missing/unexpected keys explicitly instead of treating successful loading as proof of complete parameter coverage. Loader (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`), WMRemover initialization (`task-1/.venv/Lib/site-packages/trustmark/denoise.py`)

## API and return shapes

| Entry point | Input | Return |
| --- | --- | --- |
| `tm.remove_watermark(image, WM_STRENGTH=1.0, WM_MERGE='bilinear')` | One PIL RGB image | One PIL RGB image of the original size; array representation `(H,W,3)`, `uint8` |
| `tm.removal(x)` | Float tensor `(B,3,H,W)`, normalized to `[-1,1]` | Cleaned tensor `(B,3,H,W)`, clamped to `[-1,1]` |
| `tm.removal.denoise(x)` | Same tensor | Unclamped restored tensor `(B,3,H,W)`; input plus learned correction |
| Forward hook on `tm.removal.denoise.ending` | Internal features | Correction tensor `(B,3,Hpad,Wpad)` before addition and clamping; for 256² input exactly `(B,3,256,256)` |
| `remove_watermark2` | — | **Absent from installed package and inspected upstream main; no return contract to rely on** |

These contracts are derived from installed `remove_watermark`, line 422 (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`), WMRemover.forward, line 222 (`task-1/.venv/Lib/site-packages/trustmark/denoise.py`), and KBNet_s.forward, line 178 (`task-1/.venv/Lib/site-packages/trustmark/KBNet/kbnet_s_arch.py`). Searching the entire installed package found no `remove_watermark2`; official repository default-branch code search also returned no match. This does not establish that no historical/private fork ever used that name. In particular, do not unpack `remove_watermark` as `(image,residual)` or confuse it with the encoder's internal tuple return.

## Exact installed preprocessing and reconstruction

For Q with default `concentrate_wm_region=1.0`, the implementation performs the following operations. Installed implementation, lines 189–334 and 422–438 (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`)

1. Select the processing region. If `max(width,height)/min(width,height) > 2.0`, take the centered largest square; otherwise take the full image. The comparison is strict, so an aspect ratio of exactly 2.0 uses the whole image. A nondefault concentration factor scales the central selected region using integer dimensions and floor-centered offsets.
2. Resize that region with **PIL bilinear** interpolation to 256×256. The routine does not convert image mode itself; supply `.convert('RGB')` explicitly, as in Adobe's example.
3. `ToTensor`, add a batch dimension, move to the remover device, then map `[0,1]` to `[-1,1]` with `2*x-1`.
4. Evaluate `clean256 = tm.removal(x256).clamp(-1,1)`. `WMRemover.forward` already clamps, so this outer clamp is redundant.
5. Compute `r256 = clean256 - x256`. This is a **removal correction**; input-minus-cleaned has the opposite sign.
6. Resize the residual with `torch.nn.functional.interpolate(r256, (region_H,region_W), mode=WM_MERGE)`. Default mode is bilinear. The source supplies no `align_corners` or `antialias` argument. It resizes the residual, not the full reconstructed image. After permutation, the actual NumPy shape is `(B,region_H,region_W,3)`, despite an outdated adjacent `(B,3,H,W)` comment.
7. Form `clip(original_region/127.5 - 1 + strength*r, -1,1)*127.5 + 127.5`. Q uses the requested strength unchanged; P alone multiplies it by 1.25.
8. Paste into a copy of the original converted to `uint8`, using feathering. The feather width is `min(50,max(1,int(min(region_W,region_H)*0.01)))`; each edge row/column blends with the original using `(i+1)/feather_width`. The destination is already `uint8`, so assignments quantize before the final `astype(uint8)`/PIL construction. Edge columns are processed after rows and overwrite corner blending.

There is no RGB↔YUV transform in removal inference. Unlike `encode`, the remover does **not** subtract spatial per-channel residual means. An experimental channel-centering step is therefore a distinct ablation, not faithful reproduction of removal. Saturated pixels, output clipping, uint8 truncation, and feathering change the measured difference between the final PIL output and the input. Installed encoder and remover (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`)

Recommended faithful float signal: retain both `tm.removal(x)-x` and the `.ending` output before conversion to NumPy/PIL. Compare them to measure clipping's effect. Use FP32 initially and retain the original 256² coordinates, region box, and interpolation choice. These are experiment-design recommendations derived from the code, not validated superiority claims.

## Feature hooks and spatial behavior

The following expected shapes follow from the verified Q configuration and KBNet source for a `(B,3,256,256)` input. They were calculated from layer definitions, not measured in a new model forward pass. Q YAML (`task-1/.venv/Lib/site-packages/trustmark/models/trustmark_rm_Q.yaml`), KBNet source (`task-1/.venv/Lib/site-packages/trustmark/KBNet/kbnet_s_arch.py`)

| Hook under `tm.removal.denoise` | Output `(B,C,H,W)` |
| --- | --- |
| `intro` / `encoders[0]` | `(B,32,256,256)` |
| `encoders[1]` | `(B,64,128,128)` |
| `encoders[2]` | `(B,128,64,64)` |
| `encoders[3]` | `(B,256,32,32)` |
| `middle_blks` | `(B,512,16,16)` |
| `decoders[0]` | `(B,256,32,32)` |
| `decoders[1]` | `(B,128,64,64)` |
| `decoders[2]` | `(B,64,128,128)` |
| `decoders[3]` | `(B,32,256,256)` |
| `ending` | `(B,3,256,256)` |

Ordinary PyTorch forward hooks can collect these module outputs without modifying installed files. Detach collected tensors and remove hooks after extraction. `ending` is particularly convenient: KBNet adds its output to the input before returning. The network pads only right/bottom with zeros to a multiple of 16 and crops its restored output back to the input shape. A hook sees padded dimensions if arbitrary tensor dimensions are used. Direct native-resolution inference is architecturally possible, but bypasses the released wrapper's 256² preprocessing and should be named a separate experimental mode. KBNet implementation (`task-1/.venv/Lib/site-packages/trustmark/KBNet/kbnet_s_arch.py`)

Each KBBlock includes `AdaptiveAvgPool2d(1)` in its channel-attention path. Consequently, even early features can depend on content across the full processing region. A local input perturbation can affect corrections outside that perturbation, and resized residuals cannot be assumed to preserve a sharp boundary. This is an architectural inference, not a measured receptive-field characterization. KBBlock_s, `sca` and `forward` (`task-1/.venv/Lib/site-packages/trustmark/KBNet/kbnet_s_arch.py`)

## Partial regions and experimental controls

There is no per-pixel ground-truth mask input to the installed remover. Its crop is selected geometrically, not by watermark detection. Outside a selected central crop, the public API leaves input pixels untouched. Therefore a zero correction there means **unprocessed**, not **watermark absent**. For a watermark occupying an arbitrary subregion, resizing the whole image changes the watermark's scale relative to the model. Processing a candidate crop separately changes both scale and surrounding context. Both approaches require empirical validation. Installed preprocessing and removal (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`)

The following are proposed controls, not existing ReMark guarantees:

- Compare RGB residual magnitude, signed residual channels, and a small frozen-feature probe using identical train/validation splits. Learn any thresholds only on training/validation data.
- Include clean images and matched source content where available. Removal energy on clean content measures the content-dependent baseline; it must not be silently labeled as a watermark.
- Retain both positive and negative masks and report pixel/region metrics separately from decoder success before/after removal. Failing to decode a payload does not prove recovery of its spatial support.
- Include partial regions at multiple sizes and offsets, images with aspect ratios above 2, clipping-heavy content, and compression/resize transformations relevant to the actual dataset.
- When evaluating the public wrapper, exclude or explicitly mark unprocessed pixels. If resizing the whole image instead, record that the experiment bypasses the default center-crop rule.
- Treat mean-centering, spatial smoothing, frequency filtering, repeated removal, and alternate tensor resolution as separate named ablations. Avoid reporting their output as the unmodified public removal API.

For a geometry/border control, distinguish the selected-region boundary from a real watermark boundary. At 256×256 the wrapper feather width is 2 pixels, with edge alpha 0.5 and the next row/column alpha 1.0; at 128×128 it is 1 pixel, which produces no attenuation because alpha is 1.0. Feathering follows the original selected-region dimensions, not necessarily 256. Exclude an explicitly declared band in image coordinates if testing sensitivity to this artifact. At crop corners, the final column blend uses the original image and patch directly and overwrites the previous row blend, so it is not a separable product of two alpha ramps. Installed feathering code (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`)

An oracle watermark-box crop changes more than signal dilution: it removes unrelated scene context, changes the crop aspect ratio, resizes watermark structure to the model's expected working scale, and introduces a new inference boundary. Global channel attention propagates context changes throughout the crop. Thus improved oracle-crop results cannot by themselves identify which cause helped and are an upper-bound diagnostic, not a usable detector. Fixed overlapping 128/256/384-pixel windows can test a deployable alternative; resize each window using the same declared rule, map scores back precisely, and fix overlap aggregation and boundary handling without consulting outer-test masks. If invoking the public API, each small window also receives its own quantization and feathering. If invoking the float tensor model, those wrapper effects are absent and must be labeled accordingly. Wrapper geometry (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`), KBNet global attention and padding (`task-1/.venv/Lib/site-packages/trustmark/KBNet/kbnet_s_arch.py`)

A generic blur-residual control and identically processed clean/watermarked pairs can test whether score energy reflects scene texture or generic denoising. Intermediate-feature RMS is especially vulnerable to content sensitivity because these features were not trained as watermark probabilities. Any optional fusion should include a parameter-matched control and be selected within the same inner split as thresholds. Report the chosen candidate's outer-fold performance separately from exploratory comparisons across every map, smoothing setting, threshold, window size, or fusion. Per-image oracle thresholds use target-mask information and should be separated from deployable calibration. These are methodological recommendations, not findings of the source paper.

## Upstream changes must not be conflated with installed behavior

The inspected upstream main commit is `4c60b9fdb4125fca3001f0803e6890a6b55371ac` (2026-11-11). It includes optional `loadRemover` and a **separate** `loadBBoxDetector`/`localize` path, with dedicated bounding-box weights. That detector is not the ReMark residual and is absent from installed 0.8.0. The upstream changelog lists bounding-box localization under 0.9.0. Upstream also changed the model host from the installed Netlify URL to Adobe's current host. This research did not upgrade the package or download another detector. [Pinned upstream wrapper](https://github.com/adobe/trustmark/blob/4c60b9fdb4125fca3001f0803e6890a6b55371ac/python/trustmark/trustmark.py), [pinned changelog](https://github.com/adobe/trustmark/blob/4c60b9fdb4125fca3001f0803e6890a6b55371ac/python/CHANGELOG.md)

## Reproducibility record

Local source SHA-256 values computed during this inspection:

| File under installed `trustmark/` | SHA-256 |
| --- | --- |
| `trustmark.py` | `150514038e8c037d1c512d66dc930a284e043c84a10e3e3312e2ab9dd7599d10` |
| `denoise.py` | `a77b6f9c400afe54c2d930370079e98c1bd64378142c894dcfe0c5f4704cb7ab` |
| `KBNet/kbnet_s_arch.py` | `4a5d68987a8c764c2d01f620816a8762fff52186e0abd8822fd88653653a7a4d` |

Primary-source entry points: [Adobe repository](https://github.com/adobe/trustmark), installed source (`task-1/.venv/Lib/site-packages/trustmark/trustmark.py`), [ICCV publication record](https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html), [author-hosted full ICCV paper](https://www.collomosse.com/pubs/Bui-ICCV-2025.pdf).
