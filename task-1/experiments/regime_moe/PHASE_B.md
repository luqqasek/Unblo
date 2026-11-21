# Phase B: calibration only

All thresholds fitted on inner held-out predictions and cross-fitted gates; outer predictions unchanged.

| Method | Variant | mIoU | Small AP | Small FG IoU | Large clean IoU |
|---|---|---:|---:|---:|---:|
| pixelseal | A0 | 0.5106 | 0.0768 | 0.0392 | 0.1296 |
| pixelseal | Bglobal | 0.5091 | 0.0768 | 0.0391 | 0.1257 |
| pixelseal | A1 | 0.5093 | 0.0768 | 0.0376 | 0.1294 |
| pixelseal | GateConstant | 0.4481 | 0.0168 | 0.0000 | 0.0147 |
| trustmark | A0 | 0.5073 | 0.0650 | 0.0241 | 0.1440 |
| trustmark | Bglobal | 0.5069 | 0.0650 | 0.0240 | 0.1430 |
| trustmark | A1 | 0.5075 | 0.0650 | 0.0225 | 0.1497 |
| trustmark | GateConstant | 0.4492 | 0.0239 | 0.0000 | 0.0124 |
| stegastamp | A0 | 0.4651 | 0.0493 | 0.0202 | 0.0796 |
| stegastamp | Bglobal | 0.4671 | 0.0493 | 0.0173 | 0.0918 |
| stegastamp | A1 | 0.4652 | 0.0493 | 0.0130 | 0.0725 |
| stegastamp | GateConstant | 0.4549 | 0.0190 | 0.0000 | 0.0060 |