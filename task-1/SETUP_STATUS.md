# Verified local setup — 2026-11-21

- Local Git repository initialized on main; no remote configured.
- Python 3.11.15, PyTorch 2.5.1, torchvision 0.20.1, CPU ONNX Runtime 1.23.2.
- `pip check`: no broken requirements. Five pytest tests passed.
- Official starter kit, train.zip, test.zip downloaded; archive SHA-256 hashes matched Hugging Face metadata.
- All 510 training and 1,020 test images loaded and checked at 512 x 512.
- Counts: 170 training / 340 test images per watermarking method.
- All 510 training masks downsample identically to the supplied 128 x 128 ground truth.
- PixelSeal pinned VideoSeal checkout b1bf4c0 and checkpoint loaded; 256-bit inference and feature maps passed.
- TrustMark Q encoder/decoder/remover weights downloaded; official package checksums passed; 100-bit inference and feature maps passed.
- StegaStamp ONNX original and logits export available; 100-bit inference and feature output passed.
- `outputs/baseline.pt`: 1,020 binary uint8 masks, shape [1020,128,128], 16,734,625 bytes; loaded with weights_only=True and validated against all expected names.

The baseline predicts clean everywhere and is only a format smoke check. No submission was uploaded.
CPU execution was verified locally. GPU execution and remote JURECA team configuration were not performed.
