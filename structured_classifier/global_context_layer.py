import os
import numpy as np
from structured_classifier.layer_operations import resize, normalize

from utils.utils import check_n_make_dir, save_dict


class GlobalContextLayer:
    layer_type = "GLOBAL_CONTEXT_LAYER"

    def __init__(self,
                 INPUTS,
                 name,
                 down_scale=0,
                 ):

        self.name = str(name)
        if type(INPUTS) is not list:
            INPUTS = [INPUTS]
        self.previous = INPUTS

        self.down_scale = 3

        for i, p in enumerate(self.previous):
            p.set_index(i)

        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
            "down_scale": self.down_scale,
        }

        self.index = 0

    def __str__(self):
        s = ""
        s += "{} - {}\n".format(self.layer_type, self.name)
        s += "---------------------------\n"
        for p in self.previous:
            s += "--> {}\n".format(p)
        return s

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

    def inference(self, x_input, interpolation="nearest"):
        o_h, o_w = x_input.shape[:2]

        x_img = resize(x_input,
                       height=int(o_h / 2**self.down_scale),
                       width=int(o_w / 2**self.down_scale))

        x_img = normalize(x_img)

        h, w = x_img.shape[:2]
        if len(x_img.shape) < 3:
            x_img = np.expand_dims(x_input, axis=2)
        num_f = x_img.shape[2]
        x_int_mm = np.zeros((h, w, num_f))
        x_int_pp = np.zeros((h, w, num_f))
        x_int_mp = np.zeros((h, w, num_f))
        x_int_pm = np.zeros((h, w, num_f))

        step = w + h
        for i in range(1, w - 1, 1):
            for j in range(1, h - 1, 1):
                n = w - 1 - i
                k = h - 1 - j

                i_m = i - 1
                j_m = j - 1

                n_p = n + 1
                k_p = k + 1

                x_int_mm[j, i, :] = x_img[j, i, :] / step + (
                            x_int_mm[j_m, i_m, :] + x_int_mm[j_m, i, :] + x_int_mm[j, i_m, :]) / 3
                x_int_pp[k, n, :] = x_img[k, n, :] / step + (
                            x_int_pp[k_p, n_p, :] + x_int_pp[k_p, n, :] + x_int_pp[k, n_p, :]) / 3
                x_int_mp[j, n, :] = x_img[j, n, :] / step + (
                            x_int_mp[j_m, n_p, :] + x_int_mp[j_m, n, :] + x_int_mp[j, n_p, :]) / 3
                x_int_pm[k, i, :] = x_img[k, i, :] / step + (
                            x_int_pm[k_p, i_m, :] + x_int_pm[k_p, i, :] + x_int_pm[k, i_m, :]) / 3

        x_out = [x_img, x_int_mm, x_int_mp, x_int_pm, x_int_pp]
        x_out = np.concatenate(x_out, axis=2)
        x_out = resize(x_out, height=o_h, width=o_w, interpolation="linear")
        return x_out

    def save(self, model_path):
        model_path = os.path.join(model_path, self.layer_type + "-" + self.name)
        check_n_make_dir(model_path)
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))
        for p in self.previous:
            p.save(model_path)

    def set_index(self, i):
        self.index = i
