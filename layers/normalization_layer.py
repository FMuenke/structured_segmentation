import os
import numpy as np

from utils.utils import check_n_make_dir, save_dict

from layers.layer_operations import normalize_and_standardize, normalize


class NormalizationLayer:
    layer_type = "NORMALIZATION_LAYER"

    def __init__(self, INPUTS, name, norm_option="normalize_and_standardize"):
        self.name = str(name)
        if type(INPUTS) is not list:
            INPUTS = [INPUTS]
        self.previous = INPUTS

        self.index = 0

        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
            "norm_option": norm_option
        }
        if type(norm_option) is list:
            self.norm_option = norm_option
        else:
            self.norm_option = [norm_option]

    def __str__(self):
        s = ""
        s += "{} - {}".format(self.layer_type, self.name)
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
        for opt in self.norm_option:
            x_input = self._inference(x_input, opt)
        return x_input

    def _inference(self, x_input, norm_option):
        x_img = self.get_features(x_input)
        if norm_option == "normalize_and_standardize":
            return normalize_and_standardize(x_img)
        if norm_option == "min_max_scaling":
            return normalize(x_img)
        if norm_option == "percentile_scaling_5":
            percentile = 5
            i_max = np.percentile(np.percentile(x_img, 100 - percentile, axis=0), 100 - percentile, axis=0)
            i_min = np.percentile(np.percentile(x_img, percentile, axis=0), percentile, axis=0)
            return (x_img.astype(np.float64) - i_min) / (i_max - i_min)
        if norm_option == "normalize_median":
            median_mat = np.median(np.median(x_img, axis=0), axis=0)
            x_img = x_img - median_mat
            return x_img
        if norm_option == "normalize_mean":
            median_mat = np.mean(np.mean(x_img, axis=0), axis=0)
            x_img = x_img - median_mat
            return x_img
        raise Exception("Option: {} unknown".format(self.norm_option))

    def save(self, model_path):
        model_path = os.path.join(model_path, self.layer_type + "-" + self.name)
        check_n_make_dir(model_path)
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))
        for p in self.previous:
            p.save(model_path)

    def set_index(self, i):
        self.index = i
