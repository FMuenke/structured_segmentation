import unittest
import numpy as np
from structured_segmentation.layers.input_layer.features.gabor import Gabor


class TestGabor(unittest.TestCase):
    def setUp(self):
        self.gabor = Gabor()
        self.test_image = np.random.rand(32, 32, 3)  # Example test image

    def test_build_f_maps(self):
        f_maps = self.gabor.build_f_maps(self.test_image[:, :, 0])
        self.assertEqual(f_maps.shape, (32, 32, 16))  # Assuming 4 theta values, 2 sigma values, and 2 frequency values

    def test_compute(self):
        f_maps = self.gabor._compute(self.test_image)
        self.assertEqual(len(f_maps), 3)
        self.assertEqual(f_maps[0].shape, (32, 32, 16))

if __name__ == '__main__':
    unittest.main()