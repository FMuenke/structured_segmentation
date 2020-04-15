import numpy as np

from data_structure.image_handler import ImageHandler
from data_structure.feature_map import FeatureMap


class ColorSpace:
    def __init__(self, color_space, resolution=64):
        self.color_space = color_space
        self.resolution = resolution

    def _prepare_image(self, image):
        img_h = ImageHandler(image)
        return img_h.prepare_image_for_processing(self.color_space)

    def compute(self, image):
        list_of_targets = self._prepare_image(image)
        tensor = []
        for t in list_of_targets:
            t = np.expand_dims(t, axis=2)
            tensor.append(t)
        return np.concatenate(tensor, axis=2)
