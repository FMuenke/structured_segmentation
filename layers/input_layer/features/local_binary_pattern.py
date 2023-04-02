import numpy as np
from skimage import feature

from data_structure.image_container import ImageContainer


class LocalBinaryPattern:
    def __init__(self, color_space="gray", radius=7, num_points=24, normalise=True):
        self.color_space = color_space
        self.radius = radius
        self.num_points = num_points
        self.normalise = normalise

    def _compute(self, channels):
        lbp_maps = []
        for c in channels:
            lbp_map = feature.local_binary_pattern(
                c,
                self.num_points,
                self.radius,
                method="uniform"
            )
            if self.normalise:
                lbp_map = 255 * (lbp_map - np.min(lbp_map)) / (np.max(lbp_map) - np.min(lbp_map))
            lbp_map = np.expand_dims(lbp_map, axis=2)
            lbp_maps.append(lbp_map)
        return lbp_maps

    def compute(self, image):
        img = ImageContainer(image)
        channels = img.prepare_image_for_processing(self.color_space)
        lbp_maps = self._compute(channels)
        return np.concatenate(lbp_maps, axis=2)
