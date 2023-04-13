import cv2
import numpy as np
from scipy.ndimage import convolve


def make_s_element(kernel_size, kernel_shape):
    kernel_x, kernel_y = kernel_size
    if kernel_shape == "square":
        return np.ones((kernel_y, kernel_x))
    if kernel_shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_x, kernel_y))
    if kernel_shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_x, kernel_y))
    raise Exception("Kernel-Shape option: {} not known".format(kernel_shape))


class Kernel:
    def __init__(self, kernel_size, strides, kernel_shape):
        self.kernel_x, self.kernel_y = kernel_size
        self.stride_x, self.stride_y = strides
        self.kernel_shape = kernel_shape
        s_element = make_s_element(kernel_size, kernel_shape)
        self.look_ups = []
        for i in range(self.kernel_x):
            for j in range(self.kernel_y):
                if s_element[j, i] == 1:
                    look = np.zeros((
                        int(self.stride_y * self.kernel_y),
                        int(self.stride_x * self.kernel_x),
                        1
                    ))
                    look[int(j * self.stride_y),
                         int(i * self.stride_x), 0] = 1
                    self.look_ups.append(look)

    def __str__(self):
        return "K ({},{}) ({},{})".format(
            self.kernel_x,
            self.kernel_y,
            self.stride_x,
            self.stride_y
        )

    def get_kernel(self, tensor):
        if len(tensor.shape) < 3:
            tensor = np.expand_dims(tensor, axis=2)
        num_f = tensor.shape[2]
        look_up_tensors = []
        for look in self.look_ups:
            filter_block = np.repeat(look, num_f, axis=2)
            tens = convolve(tensor, filter_block)
            look_up_tensors.append(tens)
        return np.concatenate(look_up_tensors, axis=2)
