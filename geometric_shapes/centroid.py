import numpy as np
from skimage.measure import regionprops
from skimage.draw import circle


class Centroid:
    def __init__(self):
        self.prop = None
        self.x = None

        self.num_targets = 2

    def _init_properties(self, label_map):
        prop = {}
        height, width = label_map.shape[:2]
        regions = regionprops(label_map.astype(np.int))
        if len(regions) < 1:
            prop["cy"] = 0
            prop["cx"] = 0
            return prop
        centroid = regions[0].centroid
        prop["cy"] = centroid[0] / height - 0.5
        prop["cx"] = centroid[1] / width - 0.5
        return prop

    def binarize_map(self, label_map, thresh=0.1):
        if len(label_map.shape) == 3:
            label_map = np.sum(label_map, axis=2)
        label_map[label_map < thresh] = 0
        label_map[label_map >= thresh] = 1
        return label_map

    def get_parameter(self, label_map):
        label_map = self.binarize_map(label_map)
        self.prop = self._init_properties(label_map)
        a = self.prop["cy"]
        b = self.prop["cx"]
        return a, b

    def get_label_map(self, parameters, height, width):
        label_map = np.zeros((height, width, 1))
        dia = np.sqrt(height ** 2 + width ** 2)
        cy = (parameters[0] + 0.5) * height
        cx = (parameters[1] + 0.5) * width
        rr, cc = circle(int(cy),
                        int(cx),
                        radius=0.02*dia,
                        shape=(height, width))
        label_map[rr, cc, 0] = 1
        return label_map

    def eval(self, gtr_map, pre_map):
        gtr_p = self.get_parameter(gtr_map)
        pre_p = self.get_parameter(pre_map)
        return np.sqrt((gtr_p[0] - pre_p[0])**2 + (gtr_p[1] - pre_p[1])**2)
