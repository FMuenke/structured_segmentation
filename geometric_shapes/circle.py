import numpy as np
from skimage.measure import regionprops
from skimage.draw import circle as draw_circle


class Circle:
    def __init__(self):
        self.prop = None

    def binarize_map(self, label_map, thresh=0.1):
        if len(label_map.shape) == 3:
            label_map = np.sum(label_map, axis=2)
        label_map[label_map < thresh] = 0
        label_map[label_map >= thresh] = 1
        return label_map

    def _init_properties(self, label_map):
        prop = {}
        height, width = label_map.shape[:2]
        label_map = self.binarize_map(label_map)
        regions = regionprops(label_map.astype(np.int))
        if len(regions) < 1:
            prop["cy"] = 0
            prop["cx"] = 0
            prop["radius"] = 0
            return prop
        centroid = regions[0].centroid
        prop["radius"] = regions[0].major_axis_length
        prop["cy"] = centroid[0] / height
        prop["cx"] = centroid[1] / width
        prop["radius"] = prop["radius"] / np.sqrt(height**2 + width**2)
        return prop

    def get_parameter(self, label_map):
        self.prop = self._init_properties(label_map)
        a, b = self.prop["cy"], self.prop["cx"]
        c = self.prop["radius"]
        return a, b, c

    def get_label_map(self, parameters, height, width):
        label_map = np.zeros((height, width, 1))
        dia = np.sqrt(height**2 + width**2)
        rr, cc = draw_circle(
            int(parameters[0]*height),
            int(parameters[1]*width),
            int(parameters[2]*dia),
            shape=(height, width)
        )
        label_map[rr, cc, 0] = 1
        return label_map

