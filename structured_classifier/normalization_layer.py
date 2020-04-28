import os
import numpy as np

from utils.utils import check_n_make_dir, save_dict

from structured_classifier.layer_operations import normalize_and_standardize


class NormalizationLayer:
    layer_type = "NORMALIZATION_LAYER"

    def __init__(self, INPUTS, name):
        self.name = str(name)
        if type(INPUTS) is not list:
            INPUTS = [INPUTS]
        self.previous = INPUTS

        self.index = 0

        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
        }

    def __str__(self):
        s = ""
        s += "\n{} - {}".format(self.layer_type, self.name)
        s += "\n---------------------------"
        for p in self.previous:
            s += "\n--> {}".format(p)
        return s

    def get_features(self, x_input):
        x = []
        for p in self.previous:
            x_p = p.inference(x_input, interpolation="cubic")
            if len(x_p.shape) < 3:
                x_p = np.expand_dims(x_p, axis=2)
            x.append(x_p)
        x_img = np.concatenate(x, axis=2)
        return x_img

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

    def inference(self, x_input, interpolation="nearest"):
        x_img = self.get_features(x_input)
        x_img = normalize_and_standardize(x_img)
        return x_img

    def save(self, model_path):
        model_path = os.path.join(model_path, self.layer_type + "-" + self.name)
        check_n_make_dir(model_path)
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))
        for p in self.previous:
            p.save(model_path)

    def set_index(self, i):
        self.index = i
