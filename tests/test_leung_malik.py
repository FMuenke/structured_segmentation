import unittest
import numpy as np
from structured_segmentation.layers.input_layer.features.leung_malik import LeungMalik


class TestLeungMalik(unittest.TestCase):
    def setUp(self):
        self.color_space = "rgb"
        self.lm = LeungMalik(self.color_space)
        self.test_image = np.random.randint(0, 255, size=(64, 64, 3))

    def test_constructor(self):
        self.assertIsInstance(self.lm, LeungMalik)
        self.assertEqual(self.lm.color_space, "rgb")

    def test_build_feature_tensor(self):
        feature_tensor = self.lm._build_feature_tensor(self.test_image[:, :, 0])
        self.assertEqual(feature_tensor.shape, (64, 64, 48))

    def test_compute(self):
        feature_tensors = self.lm.compute(self.test_image)
        self.assertEqual(feature_tensors.shape, (64, 64, 48*3))

if __name__ == '__main__':
    unittest.main()