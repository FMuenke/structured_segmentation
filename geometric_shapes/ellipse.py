import numpy as np
import math
from skimage.measure import regionprops
from skimage.draw import ellipse, circle, line


class Ellipse:
    def __init__(self):
        self.prop = None
        self.x = None

        self.num_targets = 8

    def _init_properties(self, label_map):
        prop = {}
        height, width = label_map.shape[:2]
        regions = regionprops(label_map.astype(np.int))
        if len(regions) == 0:
            prop["x1"] = 0
            prop["y1"] = 0
            prop["x2"] = 0
            prop["y2"] = 0
            prop["x3"], prop["y3"] = 0, 0
            prop["x4"], prop["y4"] = 0, 0
            return prop
        props = regions[0]
        y0, x0 = props.centroid[:2]
        orientation = props.orientation

        x1 = (x0 + math.cos(orientation) * 0.5 * props.minor_axis_length) / width
        y1 = (y0 - math.sin(orientation) * 0.5 * props.minor_axis_length) / height
        x2 = (x0 - math.sin(orientation) * 0.5 * props.major_axis_length) / width
        y2 = (y0 - math.cos(orientation) * 0.5 * props.major_axis_length) / height
        x3 = (x0 + math.sin(orientation) * 0.5 * props.major_axis_length) / width
        y3 = (y0 + math.cos(orientation) * 0.5 * props.major_axis_length) / height
        x4 = (x0 - math.cos(orientation) * 0.5 * props.minor_axis_length) / width
        y4 = (y0 + math.sin(orientation) * 0.5 * props.minor_axis_length) / height

        points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        x0 /= width
        y0 /= height
        for p in points:
            if p[0] >= x0 and p[1] < y0:
                prop["x1"] = p[0]
                prop["y1"] = p[1]
            if p[0] > x0 and p[1] >= y0:
                prop["x2"] = p[0]
                prop["y2"] = p[1]
            if p[0] <= x0 and p[1] > y0:
                prop["x3"] = p[0]
                prop["y3"] = p[1]
            if p[0] < x0 and p[1] <= y0:
                prop["x4"] = p[0]
                prop["y4"] = p[1]
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
        x1 = self.prop["x1"]
        y1 = self.prop["y1"]
        x2 = self.prop["x2"]
        y2 = self.prop["y2"]
        x3 = self.prop["x3"]
        y3 = self.prop["y3"]
        x4 = self.prop["x4"]
        y4 = self.prop["y4"]
        return x1, y1, x2, y2, x3, y3, x4, y4

    def mark_point(self, label_map, cx, cy, height, width):
        rr, cc = circle(int(cy * height),
                        int(cx * width),
                        radius=0.02 * np.mean([height, width]),
                        shape=(height, width))
        label_map[rr, cc, 0] = 1
        return label_map

    def mark_line(self, label_map, p0, p1, height, width):
        rr, cc = line(
            int(p0[1] * height),
            int(p0[0] * width),
            int(p1[1] * height),
            int(p1[0] * width),
        )
        label_map[rr, cc, 0] = 1
        return label_map

    def get_label_map(self, parameters, height, width):
        label_map = np.zeros((height, width, 1))

        # label_map = self.mark_point(label_map, parameters[0], parameters[1], height, width)
        # label_map = self.mark_point(label_map, parameters[2], parameters[3], height, width)
        # label_map = self.mark_point(label_map, parameters[4], parameters[5], height, width)
        # label_map = self.mark_point(label_map, parameters[6], parameters[7], height, width)

        axis_0_length = np.sqrt(
            (parameters[0] * width - parameters[4] * height) ** 2 +
            (parameters[1] * width - parameters[5] * height) ** 2)
        axis_1_length = np.sqrt(
            (parameters[2] * width - parameters[6] * height) ** 2 +
            (parameters[3] * width - parameters[7] * height) ** 2)

        ## Get Centroid
        xc1 = parameters[0] - 0.5 * (parameters[0] - parameters[4])
        yc1 = parameters[1] - 0.5 * (parameters[1] - parameters[5])
        xc2 = parameters[2] - 0.5 * (parameters[2] - parameters[6])
        yc2 = parameters[3] - 0.5 * (parameters[3] - parameters[7])

        xc = np.mean([xc1, xc2])
        yc = np.mean([yc1, yc2])

        # label_map = self.mark_point(label_map, xc, yc, height, width)

        orientation = np.arctan(
            ((parameters[0] - parameters[4]) * height) /
            ((parameters[1] - parameters[5] + 1e-5) * width))

        rr, cc = ellipse(yc*height, xc*width,
                         r_radius=axis_0_length/2,
                         c_radius=axis_1_length/2,
                         rotation=orientation,
                         shape=(height, width))

        label_map[rr, cc, 0] = 1

        return label_map

    def eval(self, gtr_map, pre_map):
        gtr_p = self.get_parameter(gtr_map)
        pre_p = self.get_parameter(pre_map)
        p1 = np.sqrt((gtr_p[0] - pre_p[0]) ** 2 + (gtr_p[1] - pre_p[1]) ** 2)
        p2 = np.sqrt((gtr_p[2] - pre_p[2]) ** 2 + (gtr_p[3] - pre_p[3]) ** 2)
        p3 = np.sqrt((gtr_p[4] - pre_p[4]) ** 2 + (gtr_p[5] - pre_p[5]) ** 2)
        p4 = np.sqrt((gtr_p[6] - pre_p[6]) ** 2 + (gtr_p[7] - pre_p[7]) ** 2)
        return np.mean([p1, p2, p3, p4])
