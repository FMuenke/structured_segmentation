import numpy as np
import math
from skimage.measure import regionprops
from skimage.draw import ellipse, circle


class EllipseAl:
    def __init__(self):
        self.prop = None
        self.x = None

        self.num_targets = 5

    def _init_properties(self, label_map):
        prop = {}
        height, width = label_map.shape[:2]
        regions = regionprops(label_map.astype(np.int))
        if len(regions) == 0:
            prop["x0"], prop["y0"] = 0, 0
            return prop
        props = regions[0]
        y0, x0 = props.centroid
        orientation = props.orientation
        prop["orientation"] = orientation
        prop["minor"] = props.minor_axis_length
        prop["major"] = props.major_axis_length
        prop["centroid"] = props.centroid
        prop["x0"] = x0
        prop["y0"] = y0

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

        x1 = self.prop["x0"]
        y1 = self.prop["y0"]

        minal = self.prop["minor"]
        maxal = self.prop["major"]

        orientation = self.prop["orientation"]

        return x1, y1, minal, maxal, orientation

    def get_label_map(self, parameters, height, width):

        label_map = np.zeros((height, width, 1))

        x_center = parameters[0]
        y_center = parameters[1]

        minal = parameters[2]
        maxal = parameters[3]

        orientation = parameters[4]

        rr, cc = ellipse(y_center, x_center,
                         r_radius = maxal/2,
                         c_radius = minal/2,
                         rotation = orientation,
                         shape = (height, width))

        label_map[rr, cc, 0] = 1

        return label_map

    def eval(self, gtr_map, pre_map):
        return 0
