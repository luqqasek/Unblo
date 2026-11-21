"""Synthetic checks without model weights, downloads or PyTorch."""
import unittest

import numpy as np

from run import METHODS, fit_priors, mean_iou, partition, predict


class BaselineTest(unittest.TestCase):
    def test_split_and_no_validation_leakage(self):
        meta = [{"method": method, "image_name": f"{method}-{i:03}"}
                for method in METHODS for i in range(170)]
        fit, val = partition(meta)
        self.assertEqual((len(fit), len(val)), (408, 102))
        self.assertFalse(set(fit) & set(val))
        self.assertEqual(set(fit) | set(val), set(range(510)))
        self.assertEqual(partition(meta), (fit, val))
        masks = np.zeros((510, 128, 128), dtype=np.uint8)
        masks[val] = 1
        priors = fit_priors(meta, masks, fit)
        predicted = predict([meta[i] for i in val], priors, "prior")
        self.assertFalse(predicted.any())
        self.assertEqual(mean_iou(predicted, masks[val]), 0)

    def test_method_specific_prior_and_tie(self):
        priors = {method: np.full((128, 128), i / 2) for i, method in enumerate(METHODS)}
        meta = [{"method": method} for method in reversed(METHODS)]
        masks = predict(meta, priors, "prior")
        self.assertEqual(masks[:, 0, 0].tolist(), [1, 1, 0])
        self.assertEqual(masks.dtype, np.uint8)
        self.assertEqual(mean_iou(masks, masks), 1)
        self.assertFalse(predict(meta, priors, "zeros").any())
        self.assertTrue(predict(meta, priors, "ones").all())


if __name__ == "__main__":
    unittest.main()
