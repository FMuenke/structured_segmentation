import cv2
import numpy as np

from data_structure.image_container import ImageContainer


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
        f_maps = []
        for c in channels:
            f_map = self.build_f_maps(c)
            f_maps.append(f_map)
        return f_maps

    def compute(self, image):
        img_h = ImageContainer(image)
        channels = img_h.prepare_image_for_processing(self.color_space)
        f_maps = self._compute(channels)
        return np.concatenate(f_maps, axis=2)


