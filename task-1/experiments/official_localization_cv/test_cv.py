import unittest
import numpy as np
from cross_validate import folds, effect
from run import METHODS, partition, fit_priors, predict


class CVTest(unittest.TestCase):
    def setUp(self):
        self.meta = [{"method": m, "image_name": f"{m}-{i:03}"} for m in METHODS for i in range(170)]

    def test_folds_cover_once_and_first_matches_original(self):
        split = list(folds(self.meta))
        np.testing.assert_array_equal(split[0][1], partition(self.meta)[1])
        seen = []
        for train, val in split:
            self.assertEqual((len(train), len(val)), (408, 102))
            self.assertFalse(set(train) & set(val))
            for m in METHODS:
                self.assertEqual(sum(self.meta[i]["method"] == m for i in val), 34)
            seen.extend(val.tolist())
        self.assertEqual(sorted(seen), list(range(510)))

    def test_validation_masks_cannot_influence_fitted_prior(self):
        train, val = next(folds(self.meta))
        masks = np.zeros((510, 128, 128), dtype=np.uint8)
        masks[val] = 1
        prior = fit_priors(self.meta, masks, train.tolist())
        self.assertFalse(predict([self.meta[i] for i in val], prior, "prior").any())

    def test_paired_effect_direction_and_folds(self):
        result = effect(np.zeros(10), np.full(10, .2), np.repeat(np.arange(5), 2))
        self.assertAlmostEqual(result["delta"], .2)
        self.assertEqual(result["positive_folds"], 5)
        np.testing.assert_allclose(result["conditional_paired_bootstrap_95"], [.2, .2])


if __name__ == "__main__":
    unittest.main()
