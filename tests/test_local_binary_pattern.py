import unittest
import numpy as np
from structured_segmentation.layers.input_layer.features.local_binary_pattern import LocalBinaryPattern

class TestLocalBinaryPattern(unittest.TestCase):
    def setUp(self):
        self.color_space = "gray"
        self.radius = 7
        self.num_points = 24
        self.normalise = True
        self.lbp = LocalBinaryPattern(self.color_space, self.radius, self.num_points, self.normalise)
        self.test_image = np.random.randint(0, 255, size=(64, 64, 3))

    def test_constructor(self):
        self.assertIsInstance(self.lbp, LocalBinaryPattern)
        self.assertEqual(self.lbp.color_space, "gray")
        self.assertEqual(self.lbp.radius, 7)
        self.assertEqual(self.lbp.num_points, 24)
        self.assertEqual(self.lbp.normalise, True)

    def test_compute(self):
        result = self.lbp.compute(self.test_image)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (64, 64, 1))

if __name__ == '__main__':
    unittest.main()