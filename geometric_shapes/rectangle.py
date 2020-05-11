import numpy as np
from skimage.measure import regionprops


class Rectangle:
    def __init__(self):
        self.prop = None

    def _init_properties(self, label_map):
        prop = {}
        height, width = label_map.shape[:2]
        label_map = self.binarize_map(label_map)
        regions = regionprops(label_map.astype(np.int))
        if len(regions) < 1:
            prop["centroid"] = [0, 0]
            prop["height"] = 1
            prop["width"] = 1
            return prop
        centroid = regions[0].centroid
        prop["cy"] = centroid[0] / height
        prop["cx"] = centroid[1] / width
        min_row, min_col, max_row, max_col = regions[0].bbox
        prop["height"] = (max_row - min_row) / height
        prop["width"] = (max_col - min_col) / width
        return prop

    def binarize_map(self, label_map, thresh=0.1):
        if len(label_map.shape) == 3:
            label_map = np.sum(label_map, axis=2)
        label_map[label_map < thresh] = 0
        label_map[label_map >= thresh] = 1
        return label_map

    def get_parameter(self, label_map):
        self.prop = self._init_properties(label_map)
        a = self.prop["cy"]
        b = self.prop["cx"]
        c = self.prop["height"]
        d = self.prop["width"]
        return a, b, c, d

    def get_label_map(self, parameters, height, width):
        label_map = np.zeros((height, width, 1))
        x1 = int(width * (parameters[1] - 0.5 * parameters[3]))
        y1 = int(height * (parameters[0] - 0.5 * parameters[2]))
        x2 = int(width * (parameters[0] + 0.5 * parameters[3]))
        y2 = int(height * (parameters[0] + 0.5 * parameters[2]))
        label_map[y1:y2, x1:x2, 0] = 1
        return label_map
