import numpy as np
import cv2
from skimage.measure import regionprops


class Rectangle:
    def __init__(self):
        self.prop = None
        self.num_targets = 4

    def _init_properties(self, label_map):
        prop = {}
        height, width = label_map.shape[:2]
        label_map = self.binarize_map(label_map)
        regions = regionprops(label_map.astype(np.int))
        if len(regions) < 1:
            prop["cy"] = 0
            prop["cx"] = 0
            prop["height"] = 0
            prop["width"] = 0
            return prop
        centroid = regions[0].centroid
        prop["cy"] = centroid[0] / height
        prop["cx"] = centroid[1] / width
        min_row, min_col, max_row, max_col = regions[0].bbox
        prop["height"] = (max_row - min_row) / height
        prop["width"] = (max_col - min_col) / width
        prop["x1"] = min_col / width - 0.5
        prop["y1"] = min_row / height - 0.5
        prop["x2"] = max_col / width - 0.5
        prop["y2"] = max_row / height - 0.5
        return prop

    def binarize_map(self, label_map, thresh=0.1):
        if len(label_map.shape) == 3:
            label_map = np.sum(label_map, axis=2)
        label_map[label_map < thresh] = 0
        label_map[label_map >= thresh] = 1
        return label_map

    def get_parameter(self, label_map):
        self.prop = self._init_properties(label_map)
        a = self.prop["x1"]
        b = self.prop["y1"]
        c = self.prop["x2"]
        d = self.prop["y2"]
        return a, b, c, d

    def get_label_map(self, parameters, height, width):
        label_map = np.zeros((height, width, 1))
        x1 = int(width * (parameters[0] + 0.5))
        y1 = int(height * (parameters[1] + 0.5))
        x2 = int(width * (parameters[2] + 0.5))
        y2 = int(height * (parameters[3] + 0.5))
        label_map[y1:y2, x1:x2, 0] = 1
        return label_map

    def eval(self, gtr_map, pre_map):
        height, width = gtr_map.shape[:2]
        gtr_param = self.get_parameter(gtr_map)
        gtr_map = self.get_label_map(gtr_param, height, width)
        gtr_map = self.binarize_map(gtr_map)
        pre_map = self.binarize_map(pre_map)
        tp_map = np.zeros(gtr_map.shape)
        tp_map[np.logical_and(gtr_map == 1, pre_map == 1)] = 1
        tp = np.sum(tp_map)

        fp_map = np.zeros(gtr_map.shape)
        fp_map[np.logical_and(gtr_map == 0, pre_map == 1)] = 1
        fp = np.sum(fp_map)

        fn_map = np.zeros(gtr_map.shape)
        fn_map[np.logical_and(gtr_map == 1, pre_map == 0)] = 1
        fn = np.sum(fn_map)

        acc = tp / (tp + fn + fp + 1e-5)
        return acc

