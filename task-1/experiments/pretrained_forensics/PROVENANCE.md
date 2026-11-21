# Pretrained forensic models: provenance and inference notes

Research date: 2026-11-21. Sources below are the authors' repositories, their linked checkpoint hosting, and upstream dependency documentation. This document records source inspection, not measured accuracy, timing, or GPU memory. No checkpoint was downloaded during this research.

## Source revisions

| Model | Repository revision inspected |
|---|---|
| TruFor | [ae54475df6f41a491d7615100feb19263dec13f7](https://github.com/grip-unina/TruFor/tree/ae54475df6f41a491d7615100feb19263dec13f7) |
| Mesorch | [33454cb6d595267033ce1109b62ad373110678db](https://github.com/scu-zjz/Mesorch/tree/33454cb6d595267033ce1109b62ad373110678db) |
| DetectiveSAM | [2ca86f12f6bab76ed693ddf105f7baa23f10b991](https://github.com/Gertlek/DetectiveSAM/tree/2ca86f12f6bab76ed693ddf105f7baa23f10b991) |

Pin downloaded code to these revisions and record actual checkpoint hashes in the experiment manifest. Upstream `main` and package releases can change.

## TruFor

### Official weights and invocation

The official inference README links [TruFor_weights.zip](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip), with published MD5 `7bee48f3476c75616c3c5721ab256ff8`. Extract `trufor.pth.tar`, then run from `TruFor_train_test`:

```powershell
python test.py -in INPUT -out OUTPUT -exp trufor_ph3 TEST.MODEL_FILE "pretrained_models/trufor.pth.tar"
```

The supplied test script accepts one file, directory, or glob and processes batch size one. Outputs include anomaly map, confidence, detection score, image dimensions, and optionally Noiseprint++. [Official inference instructions](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/TruFor_train_test/README.md).

### Input and output contract

The native loader converts PIL images to RGB, transposes HWC to CHW, and divides float pixels by **256.0**. It does not resize, crop, or apply external ImageNet normalization. [Dataset loader](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/TruFor_train_test/dataset/dataset_test.py).

The model extracts Noiseprint++ before internally applying ImageNet RGB normalization. Localization/confidence logits are bilinearly interpolated to the tensor input's height and width with `align_corners=False`. Thus native predictions retain the input coordinate frame; any experiment-side resizing must be inverted before comparison with original masks. [Model implementation](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/TruFor_train_test/lib/models/cmx/builder_np_conf.py).

Load checkpoint `['state_dict']` into `lib.utils.get_model(config)`. The forward tuple is `(pred, conf, det, npp)`. For batched output, forged probability is `softmax(pred, dim=1)[:, 1]`; confidence and detection logits require sigmoid. NPZ `imgsize` is **(H, W)**. Windows integration should prefer a wrapper: the native output path logic strips `/` but does not account for a leading `\\`; its broad exception handler can also print failures and continue. [Test implementation](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/TruFor_train_test/test.py).

### Dependencies and 8 GB GPU plan

The training conda export pins Linux packages, Python 3.7, PyTorch 1.11/CUDA 11.3, torchvision 0.12, and numerous training packages; it is unsuitable as a direct Windows environment lock. [Conda export](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/TruFor_train_test/trufor_conda.yaml).

The separate official inference Dockerfile adds only `tqdm`, `yacs>=0.1.8`, `timm>=0.5.4`, and NumPy 1.21.5 to a PyTorch 1.11 image. This is evidence that inference does not require the full MMCV training stack. Native use additionally needs image decoding and, for `TruFor_train_test/lib/utils.py`, matplotlib. [Dockerfile](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/test_docker/Dockerfile), [utilities](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/TruFor_train_test/lib/utils.py).

Practical inference recommendation, not an upstream memory guarantee: use the existing CUDA PyTorch environment, batch one, `eval()` and `inference_mode()`, starting at 512 square or a 512 maximum edge. Record any downscaling as an experimental change because it can alter forensic traces. Measure `max_memory_allocated()` before increasing resolution. The supplied phase-3 configuration disables cuDNN; changing that should be recorded if used for performance. [Configuration](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/TruFor_train_test/lib/config/trufor_ph3.yaml).

### License

TruFor has a restrictive research license: informational/nonprofit use only; unauthorized industrial or profit-oriented use is prohibited. Copies/modifications must retain notices, and public results must credit the authors. It is not MIT/Apache. Competition eligibility and any commercial use must be assessed against these actual terms. [License](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/TruFor_train_test/LICENSE.txt).

## Mesorch

### Official checkpoints

The authors link this [Google Drive folder](https://drive.google.com/drive/folders/1jwYv-S3HAZqzz0YxM9bJynBiPv-O9-6x?usp=sharing) and a [Baidu mirror](https://pan.baidu.com/s/13Byzl5kFc_vZHX8SNe6gGg?pwd=meso). IDs and byte sizes below were inspected in the public Drive folder metadata, without downloading weights:

| File | File ID | Size | Download endpoint |
|---|---|---:|---|
| mesorch-98.pth | `1PJxKteinMyaAYokKy0JhuzBnBc6bGsau` | 1,023,886,070 bytes | [Download](https://drive.google.com/uc?export=download&id=1PJxKteinMyaAYokKy0JhuzBnBc6bGsau) |
| mesorch_p-118.pth | `1uDu-wJLUA18xX0xiRoSgq2EtRm5KP4h5` | 812,736,608 bytes | [Download](https://drive.google.com/uc?export=download&id=1uDu-wJLUA18xX0xiRoSgq2EtRm5KP4h5) |

Drive may interpose its large-file confirmation page; `gdown` with the file ID can handle that. These endpoint forms are derived from the verified IDs; binary transfers were not tested. The official setup is Python 3.10, PyTorch/torchvision, `imdlbenco`, and `numpy<2`. [Official README](https://github.com/scu-zjz/Mesorch/blob/33454cb6d595267033ce1109b62ad373110678db/README.md).

### Practical integration traps

Current `mesorch.py` defines **MesorchFull**, while `test.py` imports **Mesorch**. This is a source inconsistency, not an installed-environment diagnosis. The model constructor accepts `seg_pretrain_path=None`, `conv_pretrain=False`, and `image_size=512`; no separate backbone download is needed when complete task weights load successfully. Its forward method requires an image and mask; a zero dummy mask can satisfy loss computation during prediction. The returned dictionary's `pred_mask` is already sigmoid probability, and `pred_label` is `None`. Enforce strict state loading rather than assuming an alias proves checkpoint compatibility. [Model source](https://github.com/scu-zjz/Mesorch/blob/33454cb6d595267033ce1109b62ad373110678db/mesorch.py).

The standard test shell launches four distributed workers. The Python test hardcodes `model.module.load_state_dict(ckpt['model'], strict=False)`, invokes a multiprocessing sharing strategy, and drops incomplete batches. For Windows/single GPU and a small validation subset, a direct wrapper avoids these assumptions and prevents silent sample loss. [Shell](https://github.com/scu-zjz/Mesorch/blob/33454cb6d595267033ce1109b62ad373110678db/test_mesorch_f1.sh), [test implementation](https://github.com/scu-zjz/Mesorch/blob/33454cb6d595267033ce1109b62ad373110678db/test.py).

Official evaluation resizes to 512 x 512. IMDLBenCo applies RGB ImageNet mean/std normalization after resizing; unlike TruFor, this is **external normalization**, using the usual 255 pixel scaling. Postprocessing must resize the resulting probability map to original (H, W) before bounding boxes or original-resolution metrics. The stretch has independent horizontal and vertical scales. [Transforms](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/transforms/iml_transforms.py), [dataset ordering](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/datasets/abstract_dataset.py).

Batch-one 512 inference is the initial 8 GB experiment recommendation; no official 3070 memory measurement was found. Its DCT extractors use explicit matrix operations and min/max normalization without an epsilon, so record nonfinite output checks for constant inputs. [High-frequency extractor](https://github.com/scu-zjz/Mesorch/blob/33454cb6d595267033ce1109b62ad373110678db/extractor/high_frequency_feature_extraction.py).

The repository software is MIT licensed; retain its copyright/license notice. The inspected README does not state a separate checkpoint license. [MIT license](https://github.com/scu-zjz/Mesorch/blob/33454cb6d595267033ce1109b62ad373110678db/LICENSE).

## DetectiveSAM

### Official files and hashes

The GitHub README explicitly directs weight downloads to [Gertlek/DetectiveSAM on Hugging Face](https://huggingface.co/Gertlek/DetectiveSAM). [GitHub setup](https://github.com/Gertlek/DetectiveSAM/blob/2ca86f12f6bab76ed693ddf105f7baa23f10b991/README.md).

| Artifact | Bytes | SHA-256 from Hugging Face LFS metadata |
|---|---:|---|
| [detective_sam_sota.pth](https://huggingface.co/Gertlek/DetectiveSAM/resolve/main/checkpoints/detective_sam_sota.pth) | 297,395,157 | `0c7c33c4c4ddb82459bc2b6bbfdfd1003561e2714d02163b3daffb25ede4e1cf` |
| [model_epoch22_batch999_score1.1114.pth](https://huggingface.co/Gertlek/DetectiveSAM/resolve/main/checkpoints/model_epoch22_batch999_score1.1114.pth) | 297,395,157 | `898e64f831f6e7e4e8486c0f27b139cbaa44119c449fad2570ab4f6b331ef421` |
| [sam2.1_hiera_base_plus.pt](https://huggingface.co/Gertlek/DetectiveSAM/resolve/main/sam2configs/sam2.1_hiera_base_plus.pt) | 323,606,802 | `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5` |

Source: [official hosting API tree](https://huggingface.co/api/models/Gertlek/DetectiveSAM/tree/main?recursive=true). Each inference checkpoint also needs its matching `*_params.yaml` sidecar already present in GitHub, plus `sam2configs/sam2.1_hiera_b+.yaml`. Both aliases use size 512, prompt dimension 64, downscale 16, blur/noise streams, and intensity 0.75. [Configuration resolution](https://github.com/Gertlek/DetectiveSAM/blob/2ca86f12f6bab76ed693ddf105f7baa23f10b991/detectivesam_inference/checkpoint.py), [SOTA sidecar](https://github.com/Gertlek/DetectiveSAM/blob/2ca86f12f6bab76ed693ddf105f7baa23f10b991/checkpoints/detective_sam_sota_params.yaml).

### Single-image inference and coordinates

```powershell
python -m detectivesam_inference.predict --checkpoint detective_sam_sota --target INPUT --output-dir OUTPUT --device cuda
```

Omitting source reuses target. Source content is used for display, while the network receives target and generated perturbations. Image path/mask metadata influences deterministic noise seeds; avoid supplying ground-truth masks to inference when establishing a clean evaluation protocol. [CLI](https://github.com/Gertlek/DetectiveSAM/blob/2ca86f12f6bab76ed693ddf105f7baa23f10b991/detectivesam_inference/predict.py), [sample preparation](https://github.com/Gertlek/DetectiveSAM/blob/2ca86f12f6bab76ed693ddf105f7baa23f10b991/detectivesam_inference/dataset.py).

Preprocessing uses PIL RGB, bilinear resize to 512 square, tensor conversion, and SAM2 ImageNet normalization. Outputs remain 512 square; original dimensions are not restored by the supplied runner. Keep original dimensions separately. The runtime uses inference mode and CUDA autocast, applies sigmoid to logits, and thresholds with `>` (default 0.5). Preserve its float `probability` for calibration instead of relying on display PNGs. [Runtime](https://github.com/Gertlek/DetectiveSAM/blob/2ca86f12f6bab76ed693ddf105f7baa23f10b991/detectivesam_inference/runtime.py), [SAM2 transforms](https://github.com/facebookresearch/sam2/blob/main/sam2/utils/transforms.py).

### Dependencies, Windows, and license

Requirements pin torch `>=2.5.1,<2.8`, torchvision `>=0.20.1,<0.23`, NumPy, Pillow, OpenCV headless, Hydra, `sam2`, and tqdm. [Requirements](https://github.com/Gertlek/DetectiveSAM/blob/2ca86f12f6bab76ed693ddf105f7baa23f10b991/requirements.txt). Upstream SAM2 recommends Linux/Python >=3.10 and WSL for Windows, but permits skipping its optional CUDA extension with `SAM2_BUILD_CUDA=0`. This inference path has default zero hole/sprinkle area, so it does not require those connected-component kernels. Native Windows remains a compatibility experiment. [SAM2 installation](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md), [mask postprocessing](https://github.com/facebookresearch/sam2/blob/main/sam2/utils/transforms.py).

The network processes original plus two perturbation streams, making batch one with its existing autocast the sensible initial 8 GB test. No measured RTX 3070 memory bound is claimed. [Network implementation](https://github.com/Gertlek/DetectiveSAM/blob/2ca86f12f6bab76ed693ddf105f7baa23f10b991/detectivesam_inference/models/forgerylocalizer.py).

GitHub now contains an MIT license (2026 Gert Lek). The Hugging Face card still marks the bundled release `other`, states SAM2 components are Apache-2.0, and says DetectiveSAM release terms need finalization before broader redistribution. Record this discrepancy instead of treating the weight license as settled solely from GitHub software licensing. [GitHub license](https://github.com/Gertlek/DetectiveSAM/blob/2ca86f12f6bab76ed693ddf105f7baa23f10b991/LICENSE), [weight model card](https://huggingface.co/Gertlek/DetectiveSAM/blob/main/README.md).

## Evaluation rules for this experiment

These models localize generic manipulation. Their external training sets and thresholds do not establish performance on this repository's task. Use only a declared local validation split for threshold/postprocessing selection; keep target masks out of model inputs. Report original-coordinate localization and downstream task metrics separately from each model's published paper numbers. If a 512 map is stretched back, perform bilinear probability interpolation first, then threshold; use nearest-neighbor only when resizing already-discrete masks. Preserve (row, column) = (y, x) array convention when deriving boxes.

This research document does not change the standing inverse-STN baseline or claim a leaderboard result. Actual experiment results, hardware measurements, adapters, and both dashboard languages remain the responsibility of the experiment run.
