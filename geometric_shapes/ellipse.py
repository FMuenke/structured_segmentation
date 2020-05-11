import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.draw import ellipse
from skimage.transform import rotate


class Ellipse:
    def __init__(self):
        self.prop = None
        self.x = None

    def _init_properties(self, label_map):
        prop = {}
        height, width = label_map.shape[:2]
        label_map = self.binarize_map(label_map)
        regions = regionprops(label_map.astype(np.int))
        if len(regions) < 1:
            prop["centroid"] = [0, 0]
            prop["orientation"] = 0
            prop["major_axis_length"] = 1
            prop["minor_axis_length"] = 1
            return prop
        centroid = regions[0].centroid
        prop["cy"] = centroid[0] / height
        prop["cx"] = centroid[1] / width
        prop["orientation"] = regions[0].orientation
        prop["major_axis_length"] = regions[0].major_axis_length / np.sqrt(height**2 + width**2)
        prop["minor_axis_length"] = regions[0].minor_axis_length / np.sqrt(height**2 + width**2)
        return prop

    def binarize_map(self, label_map, thresh=0.1):
        if len(label_map.shape) == 3:
            label_map = np.sum(label_map, axis=2)
        label_map[label_map < thresh] = 0
        label_map[label_map >= thresh] = 1
        return label_map

    def fit_from_map(self, label_map):
        edg = cv2.morphologyEx(label_map.astype(np.float), cv2.MORPH_GRADIENT, (2, 2))
        idx = np.where(edg > 0)
        X = np.expand_dims(idx[1], axis=1)
        Y = np.expand_dims(idx[0], axis=1)
        A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
        b = np.ones_like(X)
        return np.linalg.lstsq(A, b)[0].squeeze()

    def map_to_array(self, array_size=None, only_border=False):
        h, w = array_size
        x_coord = np.linspace(0, w, w)
        y_coord = np.linspace(0, h, h)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = self.x[0] * X_coord ** 2 + self.x[1] * X_coord * Y_coord + self.x[2] * Y_coord ** 2 + self.x[
            3] * X_coord + self.x[4] * Y_coord
        Z_coord[Z_coord < 1] = 0
        Z_coord[Z_coord >= 1] = 1
        if only_border:
            return cv2.morphologyEx(Z_coord, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
        return Z_coord

    def get_parameter(self, label_map):
        label_map = self.binarize_map(label_map)
        self.x = self.fit_from_map(label_map)
        self.prop = self._init_properties(label_map)
        a = self.prop["cy"]
        b = self.prop["cx"]
        c = self.prop["major_axis_length"]
        d = self.prop["minor_axis_length"]
        e = self.prop["orientation"]
        print("rotation:" + str(e))
        return a, b, c, d, e

    def get_label_map(self, parameters, height, width):
        label_map = np.zeros((height, width, 1))
        dia = np.sqrt(height ** 2 + width ** 2)
        rr, cc = ellipse(int(parameters[0] * height),
                         int(parameters[1] * width),
                         int(parameters[2] * dia),
                         int(parameters[3] * dia),
                         shape=(height, width))
        label_map[rr, cc, 0] = 1
        print(parameters[4])
        label_map = rotate(label_map, angle=int(parameters[4] / np.pi * 180), order=0)
        return label_map
