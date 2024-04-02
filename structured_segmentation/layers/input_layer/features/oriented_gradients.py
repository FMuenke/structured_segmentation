import cv2
import numpy as np

from structured_segmentation.layers.input_layer.features.color_space import convert_to_color_space


class OrientedGradients:
    def __init__(self, color_space="gray"):
        self.color_space = color_space

    def build_f_maps(self, image):
        # Apply the Sobel operator to compute gradients
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute edge intensity and orientation
        edge_intensity = np.sqrt(sobelx**2 + sobely**2)
        edge_orientation = np.arctan2(sobely, sobelx)
        grad = np.stack([edge_intensity, edge_orientation], axis=2)
        return grad

    def _compute(self, channels):
        return [self.build_f_maps(channels[:, :, c]) for c in range(channels.shape[2])]

    def compute(self, image):
        channels = convert_to_color_space(image, self.color_space)
        f_maps = self._compute(channels)
        return np.concatenate(f_maps, axis=2)
