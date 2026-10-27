"""Exact regression checks against the original Boundary IoU implementation."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.ndimage import binary_erosion
from clseg.metrics import SegMetrics, square_erosion


def reference(mask, radius, border_value):
    return binary_erosion(mask, structure=np.ones((3, 3)),
                          iterations=radius, border_value=border_value)


class BoundaryOptimizationTests(unittest.TestCase):
    def test_erosion_matches_reference(self):
        rng = np.random.default_rng(42)
        for shape in ((1, 1), (3, 9), (32, 48)):
            masks = [np.zeros(shape, bool), np.ones(shape, bool), rng.random(shape) > .2]
            for mask in masks:
                for radius in (1, 2, 7, 23):
                    for border in (0, 1):
                        np.testing.assert_array_equal(square_erosion(mask, radius, border),
                                                      reference(mask, radius, border))

    def test_metric_counts_match_reference(self):
        rng = np.random.default_rng(7)
        truth = rng.integers(0, 4, (2, 64, 96))
        truth[:, 20:25, 30:35] = 255
        truth[0, :15] = 255
        pred = rng.integers(0, 4, truth.shape)
        for scope in (None, [1, 3]):
            fast = SegMetrics(4, boundary_ratio=.08, eval_classes=scope)
            old = SegMetrics(4, boundary_ratio=.08, eval_classes=scope)
            fast.update(pred, truth)
            with patch('clseg.metrics.square_erosion', reference):
                old.update(pred, truth)
            for field in ('cm', 'bi', 'bu'):
                np.testing.assert_array_equal(getattr(fast, field), getattr(old, field))


if __name__ == '__main__':
    unittest.main()
