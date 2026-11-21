# Phase A: regime estimator

Outer 3-fold; models selected only by inner log loss. No segmentation-map gate features.

| Method | Selected macro F1 | Balanced accuracy | Small recall | Large recall | Log loss | ECE | Go |
|---|---:|---:|---:|---:|---:|---:|---|
| pixelseal | 0.6457 | 0.6573 | 0.6977 | 0.9647 | 0.4461 | 0.0924 | True |
| trustmark | 0.6808 | 0.6901 | 0.7556 | 0.9647 | 0.4029 | 0.0488 | True |
| stegastamp | 0.7464 | 0.7505 | 0.8000 | 0.9765 | 0.3533 | 0.0439 | True |