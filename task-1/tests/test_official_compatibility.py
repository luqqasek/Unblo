"""Verify scaffold helpers match the actual official kit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from wmloc.utils import downsample_mask as official_downsample, score
from watermark_localization.metrics import downsample_mask, mean_iou


def test_official_metric_and_grid_agreement():
    rng = np.random.default_rng(42)
    full = rng.integers(0, 2, size=(512, 512), dtype=np.uint8)
    assert np.array_equal(downsample_mask(full), official_downsample(full))
    p = rng.integers(0, 2, size=(3, 128, 128), dtype=np.uint8)
    g = rng.integers(0, 2, size=p.shape, dtype=np.uint8)
    assert np.isclose(mean_iou(p, g), score(p, g))
