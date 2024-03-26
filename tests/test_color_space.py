import unittest
import numpy as np
from structured_segmentation.layers.input_layer.features.color_space import init_color_space


class TestColorSpace(unittest.TestCase):
    def test_color_spaces(self):
        list_of_color_spaces = ["gray", "ngray", "hsv", "rgb", "RGB", "opponent", "log_geo_mean"]

        for col in list_of_color_spaces:
            image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
            color_space = init_color_space(col)
            result = color_space(image)
            self.assertEqual(result.shape[0], 100)
            self.assertEqual(result.shape[1], 100)


if __name__ == '__main__':
    unittest.main()