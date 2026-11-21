import numpy as np
import pytest
from watermark_localization.metrics import downsample_mask, mean_iou


def test_metric_empty_classes_and_opposites():
    zeros = np.zeros((128, 128), dtype=np.uint8)
    assert mean_iou(zeros, zeros) == 1
    assert mean_iou(zeros, 1 - zeros) == 0


def test_per_image_average():
    g = np.zeros((2, 128, 128), dtype=np.uint8)
    p = g.copy()
    p[1] = 1
    assert mean_iou(p, g) == 0.5


def test_half_cell_threshold():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:2, :4] = 1
    mask[:2, 4:8] = 1
    mask[1, 7] = 0
    result = downsample_mask(mask)
    assert result.shape == (128, 128)
    assert result[0, 0] == 1
    assert result[0, 1] == 0
    assert result.sum() == 1


def test_reject_nonbinary():
    with pytest.raises(ValueError):
        mean_iou([[0, 2]], [[0, 1]])
