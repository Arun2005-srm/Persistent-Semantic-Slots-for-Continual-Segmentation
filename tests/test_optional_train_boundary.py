import unittest
from unittest.mock import patch
import numpy as np
from clseg.metrics import SegMetrics


class OptionalBoundaryTests(unittest.TestCase):
    def test_confusion_scores_preserved_and_boundary_unavailable(self):
        truth = np.zeros((32, 48), dtype=np.int64)
        truth[10:20, 10:20] = 1
        truth[:2] = 255
        pred = np.roll(truth, 2, axis=1)
        pred[pred == 255] = 1
        full = SegMetrics(3)
        full.update(pred, truth)
        fast = SegMetrics(3, compute_boundary=False)
        with patch('clseg.metrics.binary_erosion', side_effect=AssertionError('Unexpected erosion')):
            fast.update(pred, truth)
        a, b = full.compute(), fast.compute()
        for key in ('accuracy', 'miou', 'iou', 'dice', 'macro_f1', 'class_iou', 'class_f1', 'class_support', 'confusion_matrix'):
            np.testing.assert_allclose(a[key], b[key], equal_nan=True)
        self.assertTrue(np.isnan(b['biou']))
        self.assertTrue(np.isnan(b['class_biou']).all())
        self.assertTrue(np.isfinite(a['biou']))


if __name__ == '__main__':
    unittest.main()
