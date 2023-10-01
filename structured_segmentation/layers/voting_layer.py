import os
import numpy as np

from structured_segmentation.utils.utils import save_dict, check_n_make_dir


class VotingLayer:
    layer_type = "VOTING_Layer"

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
        return "{} - {}".format(self.layer_type, self.name)

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

    def get_features(self, tag_3d):
        x = []
        for p in self.previous:
            x_p = p.predict(tag_3d)
            if len(x_p.shape) < 3:
                x_p = np.expand_dims(x_p, axis=2)
            x.append(x_p)
        x_img = np.concatenate(x, axis=2)
        return x_img

    def load(self, model_path):
        pass

    def save(self, model_path):
        model_path = os.path.join(model_path, self.layer_type + "-" + self.name)
        check_n_make_dir(model_path)
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))
        for p in self.previous:
            p.save(model_path)

    def set_index(self, i):
        self.index = i

    def inference(self, x_input, interpolation="nearest"):
        return self.predict(x_input)

    def predict(self, x_input):
        x_img = self.get_features(x_input)
        h, w = x_img.shape[:2]
        if len(x_img.shape) < 3:
            x_img = np.expand_dims(x_img, axis=2)
        x_img = np.array(x_img.astype(np.int))
        y_img = np.zeros((h, w))
        for i in range(w):
            for j in range(h):
                y_img[j, i] = np.bincount(x_img[j, i, :]).argmax()
        return y_img
