import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("patch_experiment", Path(__file__).with_name("run.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
rasterize = experiment.rasterize


class PatchTest(unittest.TestCase):
    def test_patch_offsets_and_empty_fallback(self):
        empty = {"boxes": [], "scores": []}
        regions = [(0, 0, 256), (256, 0, 256), (0, 256, 256), (256, 256, 256)]
        detections = [empty, empty, empty, {"boxes": [[0, 0, 128, 64]], "scores": [.9]}]
        actual = rasterize(detections, regions, 20, False)
        expected = np.zeros((128, 128), dtype=np.uint8)
        expected[64:80, 64:96] = 1
        np.testing.assert_array_equal(actual, expected)
        fallback = rasterize([empty], [(256, 256, 256)], 20, True)
        self.assertEqual(fallback.sum(), 64*64)
        self.assertTrue(fallback[64:, 64:].all())

    def test_score_ranking_and_limit(self):
        detection = {"boxes": [[0, 0, 128, 128], [256, 256, 512, 512]], "scores": [.1, .9]}
        top = rasterize([detection], [(0, 0, 512)], 1, False)
        self.assertFalse(top[:32, :32].any())
        self.assertTrue(top[64:, 64:].all())
        both = rasterize([detection], [(0, 0, 512)], 2, False)
        self.assertEqual(both.sum(), 32*32 + 64*64)


if __name__ == "__main__":
    unittest.main()
