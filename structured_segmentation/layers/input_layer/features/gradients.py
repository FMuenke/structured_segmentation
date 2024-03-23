import cv2
import numpy as np

from structured_segmentation.layers.input_layer.features.color_space import convert_to_color_space


class Gradients:
    def __init__(self, color_space="gray", orientations=32):
        self.color_space = color_space
        self.orientations = orientations

    def build_f_maps(self, image):
        gx = cv2.Sobel(np.copy(image.astype(np.uint8)), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(np.copy(image.astype(np.uint8)), cv2.CV_32F, 0, 1)
        grad = np.stack([gx, gy], axis=2)
        return grad

    def _compute(self, channels):
        return [self.build_f_maps(c) for c in channels]

    def compute(self, image):
        channels = convert_to_color_space(image, self.color_space)
        f_maps = self._compute(channels)
        return np.concatenate(f_maps, axis=2)
