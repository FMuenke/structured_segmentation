import numpy as np
from skimage import feature

from data_structure.image_handler import ImageHandler
from data_structure.feature_map import FeatureMap


class LocalBinaryPattern:
    def __init__(self, color_space="gray", radius=7, num_points=24, resolution=64):
        self.color_space = color_space
        self.radius = radius
        self.num_points = num_points
        self.resolution = resolution

    def _compute(self, channels):
        lbp_maps = []
        for c in channels:
            lbp_map = feature.local_binary_pattern(c,
                                                   self.num_points,
                                                   self.radius,
                                                   method="uniform")
            lbp_maps.append(lbp_map)
        return lbp_maps

    def _collect_histograms(self, list_of_targets, key_points):
        dc_sets = []
        for target in list_of_targets:
            f_map = FeatureMap(target)
            dc = f_map.to_descriptors_with_histogram(key_points, resolution=self.resolution)
            if dc is not None:
                dc_sets.append(dc)
        return dc_sets

    def compute(self, image):
        img = ImageHandler(image)
        channels = img.prepare_image_for_processing(self.color_space)
        lbp_maps = self._compute(channels)
        return np.array(lbp_maps)




