import os
import numpy as np

from utils.utils import check_n_make_dir, save_dict

from structured_classifier.layer_operations import normalize_and_standardize, normalize


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
        if norm_option == "normalize_median":
            median_mat = np.median(np.median(x_img, axis=0), axis=0)
            x_img = x_img - median_mat
            return x_img
        if norm_option == "normalize_mean":
            median_mat = np.mean(np.mean(x_img, axis=0), axis=0)
            x_img = x_img - median_mat
            return x_img
        if norm_option == "normalize_mean_y_axis":
            median_mat = np.mean(x_img, axis=0)
            x_img = np.add(x_img, median_mat)
            return x_img

        if norm_option == "normalize_mean_y_axis_mean":
            median_mat = np.mean(np.mean(x_img, axis=0), axis=0)
            x_img = x_img - median_mat
            median_mat = np.mean(x_img, axis=0)
            x_img = np.add(x_img, median_mat)
            return x_img

        if norm_option == "normalize_mean_axis":
            median_mat_y = np.mean(x_img, axis=0)
            median_mat_x = np.mean(x_img, axis=0)
            x_img = np.add(x_img, np.multiply(median_mat_y, 0.5))
            x_img = np.add(x_img, np.multiply(median_mat_x, 0.5))
            return x_img
        raise ValueError("Option: {} unknown".format(self.norm_option))

    def save(self, model_path):
        model_path = os.path.join(model_path, self.layer_type + "-" + self.name)
        check_n_make_dir(model_path)
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))
        for p in self.previous:
            p.save(model_path)

    def set_index(self, i):
        self.index = i
