import cv2
import numpy as np
from scipy.ndimage import convolve, rotate


def make_s_element(kernel_size, kernel_shape):
    kernel_x, kernel_y = kernel_size
    if "line_x" in kernel_shape:
        kernel = np.zeros((kernel_y, kernel_x))
        assert kernel_x % 2 != 0, "X-Dimension has to be uneven"
        middle = int((kernel_x + 1) / 2) - 1
        kernel[:, middle] = 1
        return kernel
    if "line_y" in kernel_shape:
        kernel = np.zeros((kernel_y, kernel_x))
        assert kernel_y % 2 != 0, "X-Dimension has to be uneven"
        middle = int((kernel_y + 1) / 2)
        kernel[middle, :] = 1
        return kernel
    if "square" in kernel_shape:
        return np.ones((kernel_y, kernel_x))
    if "ellipse" in kernel_shape:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_x, kernel_y))
    if "cross" in kernel_shape:
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_x, kernel_y))
    raise Exception("Kernel-Shape option: {} not known".format(kernel_shape))


class Kernel:
    def __init__(self, kernel_size, strides, kernel_shape):
        self.kernel_x, self.kernel_y = kernel_size
        self.stride_x, self.stride_y = strides
        self.kernel_shape = kernel_shape
        s_element = make_s_element(kernel_size, kernel_shape)
        if "rot" in kernel_shape:
            self.look_ups = self.build_look_ups_rot(s_element)
        else:
            self.look_ups = self.build_look_ups_reg(s_element)

    def build_look_ups_reg(self, s_element):
        look_ups = []
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
                    look_ups.append(look)
        return look_ups
    
    def build_look_ups_rot(self, base_kernel, num_orientations=8):
        kernels = []
        for angle in np.linspace(0, 180, num_orientations, endpoint=False):
            rotated_kernel = rotate(base_kernel, angle, reshape=False)
            rotated_kernel = np.expand_dims(rotated_kernel, axis=2) / np.sum(rotated_kernel)
            kernels.append(rotated_kernel)
        return kernels
        

    def __str__(self):
        return "K {} ({},{}) ({},{})".format(
            self.kernel_shape,
            self.kernel_x,
            self.kernel_y,
            self.stride_x,
            self.stride_y
        )
    
    def normalize_kernel(self, tensor):
        # Compute the L2 norm along the last dimension
        norms = np.linalg.norm(tensor, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-6
        # Normalize the vectors
        normalized_vectors = tensor / norms
        return normalized_vectors
    
    def rotate_largest_entry_to_top(self, look_up_tensors):
        # Find the index of the largest entry along the last dimension
        tensor = look_up_tensors[0]
        max_indices = np.argmax(tensor, axis=-1)
        # Create an array of indices to perform the roll operation
        roll_indices = np.expand_dims(max_indices, axis=-1) - np.arange(tensor.shape[-1])
        # Roll each vector to move the largest entry to the top
        rotated_tensors = [np.take_along_axis(tensor, roll_indices, axis=-1) for tensor in look_up_tensors]
        return rotated_tensors

    def get_kernel(self, tensor):
        if len(tensor.shape) < 3:
            tensor = np.expand_dims(tensor, axis=2)
        num_f = tensor.shape[2]
        look_up_tensors = []
        for f_n in range(num_f):
            f_map = [convolve(tensor[:, :, [f_n]], look) for look in self.look_ups]
            f_map = np.concatenate(f_map, axis=2)
            if "norm" in self.kernel_shape:
                f_map = self.normalize_kernel(f_map)
            look_up_tensors.append(f_map)
        if "rot" in self.kernel_shape:
                look_up_tensors = self.rotate_largest_entry_to_top(look_up_tensors)
        return np.concatenate(look_up_tensors, axis=2)
