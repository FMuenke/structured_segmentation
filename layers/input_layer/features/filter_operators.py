import numpy as np
from skimage.filters import frangi, hessian, sato

from data_structure.image_container import ImageContainer


class FilterOperator:
    def __init__(self, color_space="gray", filter_name="frangi"):
        self.color_space = color_space
        self.filter_name = filter_name

    def apply(self, c):
        if self.filter_name == "frangi":
            return frangi(c)
        if self.filter_name == "hessian":
            return hessian(c)
        if self.filter_name == "sato":
            return sato(c)
        raise ValueError("FilterName - {} - not known".format(self.filter_name))

    def _compute(self, channels):
        lbp_maps = []
        for c in channels:
            r_map = self.apply(c)
            r_map = r_map.astype(np.float32)
            lbp_map = np.expand_dims(r_map, axis=2)
            lbp_maps.append(lbp_map)
        return lbp_maps

    def compute(self, image):
        img = ImageContainer(image)
        channels = img.prepare_image_for_processing(self.color_space)
        lbp_maps = self._compute(channels)
        return np.concatenate(lbp_maps, axis=2)
