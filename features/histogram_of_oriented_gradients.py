import cv2
import numpy as np

from data_structure.image_handler import ImageHandler


class HistogramOfOrientedGradients:
    def __init__(self, color_space="gray", orientations=32):
        self.color_space = color_space
        self.orientations = orientations

    def build_f_maps(self, image):
        gx = cv2.Sobel(np.copy(image.astype(np.uint8)), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(np.copy(image.astype(np.uint8)), cv2.CV_32F, 0, 1)

        mag, ang = cv2.cartToPolar(gx, gy)
        # ang_sorted = ang / (2 * np.pi) * self.orientations
        # ang_sorted = ang_sorted.astype(np.int)

        # gradient_maps = []
        # for ori_idx in range(self.orientations):
        #     grad_map = np.zeros(mag.shape)
        #     grad_map[ang_sorted == ori_idx] = mag[ang_sorted == ori_idx]
        #     gradient_maps.append(grad_map)
        return np.stack([mag, ang], axis=2)

    def _compute(self, channels):
        f_maps = []
        for c in channels:
            f_map = self.build_f_maps(c)
            f_maps.append(f_map)
        return f_maps

    def compute(self, image):
        img_h = ImageHandler(image)
        channels = img_h.prepare_image_for_processing(self.color_space)
        f_maps = self._compute(channels)
        return np.concatenate(f_maps, axis=2)


