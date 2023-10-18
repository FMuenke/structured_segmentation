import unittest
import numpy as np
from structured_segmentation.layers.structured.kernel import Kernel, make_s_element


class TestKernel(unittest.TestCase):
    def setUp(self):
        self.kernel_size = (3, 3)
        self.strides = (2, 2)
        self.kernel_shape = "square"
        self.kernel = Kernel(self.kernel_size, self.strides, self.kernel_shape)
        self.test_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

    def test_constructor(self):
        self.assertIsInstance(self.kernel, Kernel)
        self.assertEqual(self.kernel.kernel_x, 3)
        self.assertEqual(self.kernel.kernel_y, 3)
        self.assertEqual(self.kernel.stride_x, 2)
        self.assertEqual(self.kernel.stride_y, 2)
        self.assertEqual(self.kernel.kernel_shape, "square")
        self.assertTrue(all(isinstance(look_up, np.ndarray) for look_up in self.kernel.look_ups))

    def test_make_s_element(self):
        s_element = make_s_element(self.kernel_size, self.kernel_shape)
        self.assertEqual(s_element.shape, (3, 3))

    def test_get_kernel(self):
        kernel_result = self.kernel.get_kernel(self.test_tensor)
        self.assertTrue(isinstance(kernel_result, np.ndarray))
        self.assertEqual(kernel_result.shape, (1, 3, 27))


if __name__ == '__main__':
    unittest.main()
