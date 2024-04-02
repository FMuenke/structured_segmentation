import unittest
import numpy as np
from structured_segmentation.layers.input_layer.input_layer import InputLayer


class TestInputLayer(unittest.TestCase):
    def test_feature_spaces(self):
        list_of_possible_feature_spaces = [
            "color",
            "lm",
            "lbp",
            "gradient",
            "oriented_gradient",
            "laplacian",
            "gabor",
            "gaussian",
        ]
        for feat in list_of_possible_feature_spaces:
            test_img = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
            input_layer = InputLayer(name="Test1", features_to_use="gray-{}".format(feat))
            result = input_layer.inference(test_img)
            self.assertIsNotNone(result)

        for feat in list_of_possible_feature_spaces:
            test_img = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
            input_layer = InputLayer(name="Test1", features_to_use="hsv-{}".format(feat))
            result = input_layer.inference(test_img)
            self.assertIsNotNone(result)

    def test_resize_image(self):
        test_img = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
        input_layer = InputLayer(name="Test1", features_to_use="gray-gradient", initial_down_scale=1)
        result = input_layer.inference(test_img)
        self.assertEqual(result.shape[0], 50)
        self.assertEqual(result.shape[1], 50)

        test_img = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
        input_layer = InputLayer(name="Test1", features_to_use="gray-gradient", width=40, height=70)
        result = input_layer.inference(test_img)
        self.assertEqual(result.shape[0], 70)
        self.assertEqual(result.shape[1], 40)

if __name__ == '__main__':
    unittest.main()