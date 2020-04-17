import numpy as np
from scipy.ndimage import convolve

from features.color_space import ColorSpace
from features.local_binary_pattern import LocalBinaryPattern
from features.leung_malik import LeungMalik


class FeatureExtractor:
    def __init__(self, opt):
        self.features_to_use = opt["features_to_use"]
        self.look_up_window = opt["look_up_window"]
        self.look_up_window_gradient = opt["look_up_window_gradient"]
        self.look_ups = []
        if self.look_up_window is not None:
            for i in range(self.look_up_window):
                for j in range(self.look_up_window):
                    if i == j:
                        if i == (self.look_up_window - 1) / 2:
                            continue
                    look = np.zeros((self.look_up_window, self.look_up_window, 1))
                    look[j, i, 0] = 1
                    self.look_ups.append(look)

        if self.look_up_window_gradient is not None:
            for i in range(self.look_up_window_gradient):
                for j in range(self.look_up_window_gradient):
                    if i == j:
                        if i == (self.look_up_window_gradient - 1) / 2:
                            continue
                    look = np.zeros((self.look_up_window_gradient, self.look_up_window_gradient, 1))
                    look[j, i, 0] = -1
                    look[int((self.look_up_window_gradient - 1) / 2),
                         int((self.look_up_window_gradient - 1) / 2),
                         0] = 1
                    self.look_ups.append(look)

    def build_feature_tensor(self, image):
        tensors = []
        for f_type in self.features_to_use:
            if "raw" in f_type:
                tensors.append(image)
            if "lbp" in f_type:
                color_space, descriptor_type = f_type.split("-")
                lbp = LocalBinaryPattern(color_space=color_space)
                tensors.append(lbp.compute(image))
            if "color" in f_type:
                color_space, descriptor_type = f_type.split("-")
                col = ColorSpace(color_space=color_space)
                f = col.compute(image)
                tensors.append(f)
            if "lm" in f_type:
                color_space, descriptor_type = f_type.split("-")
                lm = LeungMalik(color_space=color_space)
                f = lm.compute(image)
                tensors.append(f)

        return np.concatenate(tensors, axis=2)

    def look_up(self, tensor):
        num_f = tensor.shape[2]
        look_up_tensors = [tensor]

        for look in self.look_ups:
            filter_block = np.repeat(look, num_f, axis=2)
            tens = convolve(tensor, filter_block)
            look_up_tensors.append(tens)
        return np.concatenate(look_up_tensors, axis=2)

    def extract(self, image):
        tensor = self.build_feature_tensor(image)
        tensor = self.look_up(tensor)
        return tensor

