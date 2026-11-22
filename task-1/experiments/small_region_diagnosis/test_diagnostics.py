"""Guard diagnostic conclusions against threshold/tie, connectivity and crop bugs."""
import numpy as np
from wmloc.utils import miou_image
from .common import ious
from .oof import exact_oracle, hysteresis
from .decoder_probe import background, centered


def test_oracle_matches_exhaustive_with_ties_and_single_class_images():
    p = np.array([[[.1, .8], [.8, .2]], [[.4, .1], [.4, .8]], [[.2, .2], [.8, .8]]], np.float32)
    y = np.array([[[0, 1], [0, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]]], bool)
    thresholds = np.r_[np.unique(p), np.float32(1.1)]
    for metric in ("fg_iou", "miou"):
        result = exact_oracle(p, y, metric)
        expected = max(np.mean([ious(a >= t, b)[metric] for a, b in zip(p, y)]) for t in thresholds)
        actual = np.mean([ious(a >= result["threshold"], b)[metric] for a, b in zip(p, y)])
        assert abs(result["objective"]-expected) < 1e-12
        assert abs(actual-expected) < 1e-12
    result = exact_oracle(p[:1], np.zeros_like(y[:1]))
    assert not (p[0] >= result["threshold"]).any()


def test_official_metric_and_hysteresis_connectivity():
    p = np.array([[.8, .3, 0, .3], [0, .3, 0, .3], [0, 0, .3, 0]])
    expected = np.array([[1, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]], bool)
    assert np.array_equal(hysteresis(p, .2, .7), expected)  # eight-connected bridge
    assert not hysteresis(p, .2, .9).any()
    assert np.array_equal(hysteresis(p, .5, .5), p >= .5)
    assert abs(ious(p >= .5, expected)["miou"]-miou_image(p >= .5, expected)) < 1e-12


def test_crop_controls_are_clean_and_centered_without_padding():
    mask = np.zeros((512, 512), bool); mask[:90, :100] = True
    x0, y0, x1, y1 = background(mask, 128, np.random.default_rng(42))
    assert not mask[y0:y1, x0:x1].any() and (x1-x0, y1-y0) == (128, 128)
    assert centered([0, 0, 30, 40], 128) == [0, 0, 128, 128]
    assert background(np.ones_like(mask), 128, np.random.default_rng(42)) is None
