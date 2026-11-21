import unittest
import numpy as np
from evaluate import box_mask


class BoxAdapterTest(unittest.TestCase):
    def test_no_boxes_and_clipping(self):
        self.assertFalse(box_mask(None).any())
        self.assertTrue(box_mask([[-1, -1, 2, 2]]).all())

    def test_xy_order_and_union(self):
        mask = box_mask([[0, 0, 0.5, 0.25], [0.5, 0.75, 1, 1]])
        expected = np.zeros((128, 128), dtype=np.uint8)
        expected[:32, :64] = 1
        expected[96:, 64:] = 1
        np.testing.assert_array_equal(mask, expected)

    def test_half_cell_rule(self):
        mask = box_mask([[0, 0, 4/512, 2/512]])
        self.assertEqual(mask.sum(), 1)
        self.assertEqual(mask[0, 0], 1)


if __name__ == "__main__":
    unittest.main()
