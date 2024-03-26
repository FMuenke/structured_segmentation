import numpy as np
import cv2
import os
from structured_segmentation.layers.input_layer.features.color_space import ColorSpace
from structured_segmentation.layers.input_layer.features.local_binary_pattern import LocalBinaryPattern
from structured_segmentation.layers.input_layer.features.leung_malik import LeungMalik
from structured_segmentation.layers.input_layer.features.gradients import Gradients
from structured_segmentation.layers.input_layer.features.laplacian import Laplacian
from structured_segmentation.layers.input_layer.features.gaussian import Gaussian
from structured_segmentation.layers.input_layer.features.gabor import Gabor

from structured_segmentation.utils.utils import check_n_make_dir, save_dict


def resize_image(image, height, width, down_scale):
    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    if width is not None and height is None:
        h, w = image.shape[:2]
        s = width / w
        h_new = int(h * s)
        image = cv2.resize(image, (width, h_new), interpolation=cv2.INTER_CUBIC)

    if height is not None and width is None:
        h, w = image.shape[:2]
        s = height / h
        w_new = int(w * s)
        image = cv2.resize(image, (w_new, height), interpolation=cv2.INTER_CUBIC)

    if down_scale is not None:
        height, width = image.shape[:2]
        new_height = int(height / 2 ** down_scale)
        new_width = int(width / 2 ** down_scale)
        assert new_height > 2 or new_width > 2, "ERROR: Image was scaled too small Height, Width: {}, {}".format(
            new_height, new_width)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return image


def init_features(config):
    color_space, feature_type = config.split("-")
    feature_options = {
            "color": ColorSpace,
            "lm": LeungMalik,
            "lbp": LocalBinaryPattern,
            "gradient": Gradients,
            "laplacian": Laplacian,
            "gabor": Gabor,
            "gaussian": Gaussian,
        }
    return feature_options[feature_type](color_space=color_space)


class InputLayer:
    layer_type = "INPUT_LAYER"

    def __init__(self, name, features_to_use, height=None, width=None, initial_down_scale=None):
        self.name = name
        self.features_to_use = features_to_use
        self.height = height
        self.width = width
        self.down_scale = initial_down_scale

        self.index = 0

        self.features = init_features(features_to_use)

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

    def inference(self, image):
        image = resize_image(image, self.height, self.width, self.down_scale)
        return self.features.compute(image)
        

    def set_index(self, i):
        self.index = i
