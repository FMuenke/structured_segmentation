import cv2
import numpy as np

from data_structure.image_container import ImageContainer


class Laplacian:
    def __init__(self, color_space="gray"):
        self.color_space = color_space

    def build_f_maps(self, image):
        l03 = cv2.Laplacian(np.copy(image.astype(np.uint8)), -1, ksize=3)
        l05 = cv2.Laplacian(np.copy(image.astype(np.uint8)), -1, ksize=5)
        l09 = cv2.Laplacian(np.copy(image.astype(np.uint8)), -1, ksize=9)
        l17 = cv2.Laplacian(np.copy(image.astype(np.uint8)), -1, ksize=17)
        l21 = cv2.Laplacian(np.copy(image.astype(np.uint8)), -1, ksize=21)
        l31 = cv2.Laplacian(np.copy(image.astype(np.uint8)), -1, ksize=31)
        grad = np.stack([l03, l05, l09, l17, l21, l31], axis=2)
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


