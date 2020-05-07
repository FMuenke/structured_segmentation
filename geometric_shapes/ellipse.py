import cv2
import numpy as np
import math
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.draw import line


class Ellipse:
    def __init__(self, label_map):
        label_map = label_map.astype(np.float)
        self.array_size = label_map.shape[:2]
        self.x = self.fit_from_map(label_map)

        self.prop = self._init_properties()

    def __str__(self):
        return "Ellipse defined by {0:.3}x^2+{1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1".format(self.x[0],
                                                                                            self.x[1],
                                                                                            self.x[2],
                                                                                            self.x[3],
                                                                                            self.x[4])

    def _init_properties(self):
        prop = {}
        label_map = self.map_to_array()
        regions = regionprops(label_map.astype(np.int))
        if len(regions) < 1:
            prop["centroid"] = [0, 0]
            prop["orientation"] = 0
            prop["major_axis_length"] = 1
            prop["minor_axis_length"] = 1
            return prop
        prop["centroid"] = regions[0].centroid
        prop["orientation"] = regions[0].orientation
        prop["major_axis_length"] = regions[0].major_axis_length
        prop["minor_axis_length"] = regions[0].minor_axis_length
        return prop

    def binarize_map(self, label_map, thresh=0.1):
        if len(label_map.shape) == 3:
            label_map = np.sum(label_map, axis=2)
        label_map[label_map < thresh] = 0
        label_map[label_map >= thresh] = 1
        return label_map

    def select_one_object(self, label_map):
        label_objects, nb_labels = ndi.label(label_map)
        sizes = np.bincount(label_objects.ravel())
        if len(sizes) > 2:
            mask_sizes = sizes > sorted(sizes)[-2] - 1
            mask_sizes[0] = 0
            return mask_sizes[label_objects]
        return label_map

    def fit_from_map(self, label_map):
        label_map = self.binarize_map(label_map)
        label_map = self.select_one_object(label_map)
        edg = cv2.morphologyEx(label_map.astype(np.float), cv2.MORPH_GRADIENT, (2, 2))
        idx = np.where(edg > 0)
        X = np.expand_dims(idx[1], axis=1)
        Y = np.expand_dims(idx[0], axis=1)
        A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
        b = np.ones_like(X)
        return np.linalg.lstsq(A, b)[0].squeeze()

    def map_to_array(self, array_size=None, only_border=False):
        if array_size is None:
            array_size = self.array_size
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

    def generate_gaussian(self, array_size, cx, cy, sigma):
        x, y = np.meshgrid(np.linspace(0, array_size[1], array_size[1]), np.linspace(0, array_size[0], array_size[0]))
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
        return g

    def build_label_map(self, array_size=None):
        if array_size is None:
            array_size = self.array_size

        x0 = self.prop["centroid"][1]
        y0 = self.prop["centroid"][0]
        centroid = self.generate_gaussian(array_size, x0, y0, self.prop["minor_axis_length"] / 8)
        border = self.map_to_array(array_size, only_border=True)

        max1 = np.clip(int(x0 + math.sin(self.prop['orientation']) * 0.5 * self.prop['major_axis_length']), 0,
                       array_size[1] - 1)
        may1 = np.clip(int(y0 + math.cos(self.prop['orientation']) * 0.5 * self.prop['major_axis_length']), 0,
                       array_size[0] - 1)

        max2 = np.clip(int(x0 - math.sin(self.prop['orientation']) * 0.5 * self.prop['major_axis_length']), 0,
                       array_size[1] - 1)
        may2 = np.clip(int(y0 - math.cos(self.prop['orientation']) * 0.5 * self.prop['major_axis_length']), 0,
                       array_size[0] - 1)

        mix1 = np.clip(int(x0 + math.cos(self.prop['orientation']) * 0.5 * self.prop['minor_axis_length']), 0,
                       array_size[1] - 1)
        miy1 = np.clip(int(y0 - math.sin(self.prop['orientation']) * 0.5 * self.prop['minor_axis_length']), 0,
                       array_size[0] - 1)

        mix2 = np.clip(int(x0 - math.cos(self.prop['orientation']) * 0.5 * self.prop['minor_axis_length']), 0,
                       array_size[1] - 1)
        miy2 = np.clip(int(y0 + math.sin(self.prop['orientation']) * 0.5 * self.prop['minor_axis_length']), 0,
                       array_size[0] - 1)

        ma_min_axis = np.zeros(array_size)
        rr, cc = line(int(may1), int(max1), int(may2), int(max2))
        ma_min_axis[rr, cc] = 1
        rr, cc = line(int(miy1), int(mix1), int(miy2), int(mix2))
        ma_min_axis[rr, cc] = 1
        out = np.stack([centroid, border, ma_min_axis], axis=2)
        return out

    def compare_to_map(self, label_map):
        area = label_map.shape[0] * label_map.shape[1]
        ellipse_map = self.map_to_array(label_map.shape[:2])
        label_map = self.binarize_map(label_map)
        combined = ellipse_map + label_map
        intersection = np.zeros(combined.shape)
        intersection[combined == 2] = 1
        union = np.zeros(combined.shape)
        union[combined > 0] = 1
        I = np.sum(intersection) / area
        U = np.sum(union) / area
        IoU = I / (U + 1e-6)
        return IoU

    def compare_to_ellipse(self, ellipse):
        center_diff = np.sqrt((self.prop["centroid"][0] - ellipse.prop["centroid"][0]) ** 2
                              + (self.prop["centroid"][1] - ellipse.prop["centroid"][1]) ** 2)

        orientation_diff = np.square(self.prop["orientation"] - self.prop["orientation"])
        minor_axis_length_diff = np.square(self.prop["minor_axis_length"] - ellipse.prop["minor_axis_length"])
        major_axis_length_diff = np.square(self.prop["major_axis_length"] - ellipse.prop["major_axis_length"])

        return [center_diff, orientation_diff, major_axis_length_diff, minor_axis_length_diff]