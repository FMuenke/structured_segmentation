import numpy as np
import os

from utils.utils import check_n_make_dir, save_dict


class BottleNeckLayer:
    layer_type = "BOTTLE_NECK_LAYER"

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
        s += "{} - {}".format(self.layer_type, self.name)
        return s

    def get_features(self, x_input):
        x = []
        for p in self.previous:
            x_p = p.predict(x_input)
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
        return x_img

    def load(self, model_folder):
        pass

    def predict(self, x_input, interpolation="nearest"):
        x_img = self.get_features(x_input)
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
