import numpy as np
import cv2


class Arbitrary:
    def __init__(self):
        self.prop = None

    def binarize_map(self, label_map, thresh=0.1):
        if len(label_map.shape) == 3:
            label_map = np.sum(label_map, axis=2)
        label_map[label_map < thresh] = 0
        label_map[label_map >= thresh] = 1
        return label_map

    def get_parameter(self, label_map, global_kernel):
        label_map = self.binarize_map(label_map)
        label_map = cv2.resize(label_map, (global_kernel[1], global_kernel[0]), interpolation=cv2.INTER_NEAREST)
        return np.reshape(label_map, (1, -1))

    def get_label_map(self, parameters, height, width, global_kernel):
        label_map = np.reshape(parameters, (global_kernel[0], global_kernel[1], 1))
        label_map = cv2.resize(label_map, (width, height))
        return label_map

