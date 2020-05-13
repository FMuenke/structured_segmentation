import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.draw import ellipse


class Ellipse:
    def __init__(self):
        self.prop = None
        self.x = None

    def _init_properties(self, label_map):
        prop = {}
        height, width = label_map.shape[:2]
        regions = regionprops(label_map.astype(np.int))
        if len(regions) < 1:
            prop["cy"] = 0
            prop["cx"] = 0
            prop["orientation"] = 0
            prop["height"] = 0
            prop["width"] = 0
            prop["major_axis_length"] = 0
            prop["minor_axis_length"] = 0
            return prop
        centroid = regions[0].centroid
        prop["cy"] = centroid[0] / height
        prop["cx"] = centroid[1] / width
        prop["orientation"] = regions[0].orientation
        min_row, min_col, max_row, max_col = regions[0].bbox
        prop["height"] = (max_row - min_row) / height
        prop["width"] = (max_col - min_col) / width
        prop["major_axis_length"] = regions[0].major_axis_length / np.sqrt(height**2 + width**2)
        prop["minor_axis_length"] = regions[0].minor_axis_length / np.sqrt(height**2 + width**2)
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
        c = self.prop["height"]
        d = self.prop["width"]
        e = self.prop["orientation"]
        return a, b, c, d, e

    def get_label_map(self, parameters, height, width):
        label_map = np.zeros((height, width, 1))
        dia = np.sqrt(height ** 2 + width ** 2)
        orientation = parameters[4]
        cy = parameters[0] * height
        cx = parameters[1] * width
        axis_0_length = parameters[2] * height / np.cos(-orientation)
        axis_1_length = parameters[3] * width / np.cos(np.pi - orientation)
        major_axis_length = max(axis_1_length, axis_0_length)
        minor_axis_length = min(axis_1_length, axis_0_length)
        rr, cc = ellipse(int(cy),
                         int(cx),
                         int(major_axis_length / 2),
                         int(minor_axis_length / 2),
                         rotation=orientation,
                         shape=(height, width))
        label_map[rr, cc, 0] = 1
        return label_map
