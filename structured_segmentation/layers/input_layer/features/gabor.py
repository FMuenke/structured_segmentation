import numpy as np

from skimage.filters import gabor_kernel
from scipy.ndimage import convolve

from structured_segmentation.layers.input_layer.features.color_space import convert_to_color_space


class Gabor:
    def __init__(self, color_space="gray"):
        self.color_space = color_space

        self.kernels = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                    self.kernels.append(kernel)

    def build_f_maps(self, image):
        filtered = [convolve(image, kernel) for kernel in self.kernels]
        return np.stack(filtered, axis=2)

    def _compute(self, channels):
        f_maps = [self.build_f_maps(channels[:, :, c]) for c in range(channels.shape[2])]
        return f_maps

    def compute(self, image):
        channels = convert_to_color_space(image, self.color_space)
        f_maps = self._compute(channels)
        return np.concatenate(f_maps, axis=2)
