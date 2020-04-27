import numpy as np
import cv2
import os
from features.color_space import ColorSpace
from features.local_binary_pattern import LocalBinaryPattern
from features.leung_malik import LeungMalik

from utils.utils import check_n_make_dir, save_dict


class InputLayer:
    layer_type = "INPUT_LAYER"

    def __init__(self, name, features_to_use, height=None, width=None):
        self.name = name
        if type(features_to_use) is not list:
            features_to_use = [features_to_use]
        self.features_to_use = features_to_use
        self.height = height
        self.width = width

        self.index = 0

        self.opt = {
            "name": name,
            "layer_type": self.layer_type,
            "features_to_use": self.features_to_use,
            "height": height,
            "width": width,
        }

    def __str__(self):
        return "{}-{}-{}".format(self.layer_type, self.name, self.features_to_use)

    def fit(self, train_tags, validation_tags, reduction_factor):
        pass

    def save(self, model_path):
        model_path = os.path.join(model_path, self.layer_type + "-" + self.name)
        check_n_make_dir(model_path)
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))

    def load(self, model_path):
        pass

    def inference(self, image, interpolation="nearest"):
        if self.height is not None and self.width is not None:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        tensors = []
        for f_type in self.features_to_use:
            if "raw" in f_type:
                if len(image.shape) < 3:
                    image = np.expand_dims(image, axis=2)
                tensors.append(image)
            if "lbp" in f_type:
                color_space, descriptor_type = f_type.split("-")
                lbp = LocalBinaryPattern(color_space=color_space)
                f = lbp.compute(image)
                tensors.append(f)
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

    def set_index(self, i):
        self.index = i
