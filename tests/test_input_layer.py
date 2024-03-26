import unittest
from unittest.mock import MagicMock, patch
from structured_segmentation.layers.input_layer.input_layer import InputLayer


class TestInputLayer(unittest.TestCase):
    def setUp(self):
        self.name = "input_layer"
        self.features_to_use = "gray-color"
        self.height = 100
        self.width = 100
        self.initial_down_scale = 0.5
        self.input_layer = InputLayer(self.name, self.features_to_use, self.height, self.width, self.initial_down_scale)

    def test_constructor(self):
        self.assertEqual(self.input_layer.name, self.name)
        self.assertEqual(self.input_layer.features_to_use, self.features_to_use)
        self.assertEqual(self.input_layer.height, self.height)
        self.assertEqual(self.input_layer.width, self.width)
        self.assertEqual(self.input_layer.down_scale, self.initial_down_scale)
        self.assertEqual(self.input_layer.index, 0)

    def test_str(self):
        expected_str = "INPUT_LAYER-input_layer-gray-color"
        self.assertEqual(str(self.input_layer), expected_str)

    def test_inference(self):
        image = MagicMock()
        resized_image = MagicMock()
        self.input_layer.features.compute = MagicMock(return_value="computed_features")
        resize_image_mock = MagicMock(return_value=resized_image)
        with patch("structured_segmentation.layers.input_layer.input_layer.resize_image", resize_image_mock):
            result = self.input_layer.inference(image)
        resize_image_mock.assert_called_once_with(image, self.height, self.width, self.initial_down_scale)
        self.input_layer.features.compute.assert_called_once_with(resized_image)
        self.assertEqual(result, "computed_features")

    def test_set_index(self):
        index = 5
        self.input_layer.set_index(index)
        self.assertEqual(self.input_layer.index, index)


if __name__ == '__main__':
    unittest.main()