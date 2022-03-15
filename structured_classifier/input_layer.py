import numpy as np
import cv2
import os
from features.color_space import ColorSpace
from features.local_binary_pattern import LocalBinaryPattern
from features.leung_malik import LeungMalik
from features.filter_operators import FilterOperator
from features.gradients import Gradients
from features.laplacian import Laplacian
from features.gaussian import Gaussian

from utils.utils import check_n_make_dir, save_dict


class InputLayer:
    layer_type = "INPUT_LAYER"

    def __init__(self, name, features_to_use, height=None, width=None, initial_down_scale=None):
        self.name = name
        if type(features_to_use) is not list:
            features_to_use = [features_to_use]
        self.features_to_use = features_to_use
        self.height = height
        self.width = width
        self.down_scale = initial_down_scale

        self.index = 0

        self.opt = {
            "name": name,
            "layer_type": self.layer_type,
            "features_to_use": self.features_to_use,
            "height": height,
            "width": width,
            "down_scale": initial_down_scale,
        }

    def __str__(self):
        return "{}-{}-{}".format(self.layer_type, self.name, self.features_to_use)

    def fit(self, train_tags, validation_tags):
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

        if self.width is not None and self.height is None:
            h, w = image.shape[:2]
            s = self.width / w
            h_new = int(h * s)
            image = cv2.resize(image, (self.width, h_new), interpolation=cv2.INTER_CUBIC)

        if self.height is not None and self.width is None:
            h, w = image.shape[:2]
            s = self.height / h
            w_new = int(w * s)
            image = cv2.resize(image, (w_new, self.height), interpolation=cv2.INTER_CUBIC)

        if self.down_scale is not None:
            height, width = image.shape[:2]
            new_height = int(height / 2**self.down_scale)
            new_width = int(width / 2**self.down_scale)
            assert new_height > 2 or new_width > 2, "ERROR: Image was scaled too small Height, Width: {}, {}".format(
                new_height, new_width)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        tensors = []
        for f_type in self.features_to_use:
            if "raw" in f_type:
                if len(image.shape) < 3:
                    image = np.expand_dims(image, axis=2)
                tensors.append(image)
            if "lbp" in f_type:
                color_space, descriptor_type = f_type.split("-")
                if descriptor_type == "lbp":
                    lbp = LocalBinaryPattern(color_space=color_space)
                else:
                    name, radius, num_points = descriptor_type.split(".")
                    lbp = LocalBinaryPattern(
                        color_space=color_space,
                        radius=int(radius),
                        num_points=int(num_points)
                    )
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
            if "filter" in f_type:
                color_space, filter_type, descriptor_type = f_type.split("-")
                fo = FilterOperator(color_space=color_space, filter_name=filter_type)
                f = fo.compute(image)
                tensors.append(f)
            if "gradient" in f_type:
                color_space, descriptor_type = f_type.split("-")
                gr = Gradients(color_space=color_space)
                f = gr.compute(image)
                tensors.append(f)
            if "laplacian" in f_type:
                color_space, descriptor_type = f_type.split("-")
                gr = Laplacian(color_space=color_space)
                f = gr.compute(image)
                tensors.append(f)
            if "gaussian" in f_type:
                color_space, descriptor_type = f_type.split("-")
                gr = Gaussian(color_space=color_space)
                f = gr.compute(image)
                tensors.append(f)
        data = np.concatenate(tensors, axis=2)
        return data

    def set_index(self, i):
        self.index = i
