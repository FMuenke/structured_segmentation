import cv2
import os
import logging
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from structured_segmentation.layers.structured.kernel import Kernel
from structured_segmentation.learner.internal_encoder import InternalEncoder
from structured_segmentation.layers.layer_operations import resize
from structured_segmentation.utils.utils import check_n_make_dir, save_dict, load_dict


class EmbeddingLayer:
    layer_type = "ENCODER_LAYER"

    def __init__(self,
                 INPUTS,
                 name,
                 summary="avg",
                 enc="pca",
                 data_reduction=0):
        
        self.name = str(name)
        if type(INPUTS) is not list:
            INPUTS = [INPUTS]
        self.previous = INPUTS

        assert 0 <= data_reduction < 1, "DataReduction should be inbetween [0, 1)"

        self.index = 0

        for i, p in enumerate(self.previous):
            p.set_index(i)
        
        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
            "summary": summary,
        }
        self.is_fitted = False

        self.enc = InternalEncoder(opt={"type": enc})
        self.enc.new()

    def __str__(self):
        return "{} - {} - {} - summary: {}".format(
            self.layer_type,
            self.name,
            self.enc,
            self.opt["summary"]
        )
    
    def summarize(self, x):
        if self.opt["summary"] == "avg":
            x = np.reshape(x, (-1, x.shape[2]))
            return np.mean(x, axis=0)
        
        raise Exception("Unknown SummaryFunc: {}".format(self.opt["summary"]))


    def inference(self, x_input, interpolation="nearest"):
        x_img = self.get_features(x_input)
        x_height, x_width = x_img.shape[:2]
        x_img = np.reshape(x_img, (x_height * x_width, -1))
        y_img = self.enc.transform(x_img)
        return y_img

    def predict(self, x_input):
        x_img = self.get_features(x_input)
        y_img = self.enc.transform(x_img)
        return y_img

    def get_features(self, x_input):
        x = []
        for p in self.previous:
            x_p = p.inference(x_input)
            if len(x_p.shape) < 3:
                x_p = np.expand_dims(x_p, axis=2)
            x.append(x_p)
        x = np.concatenate(x, axis=2)
        return self.summarize(x)

    def get_x_y(self, tag_set, reduction_factor=0):
        x = None
        for t in tqdm(tag_set):
            x_img = t.load_x()
            x_img = self.get_features(x_img)
            x_img = np.expand_dims(x_img, axis=0)
            print(x_img.shape)

            if x is None:
                x = x_img
            else:
                x = np.append(x, x_img, axis=0)
        x = x.astype(np.float32)
        print(x.shape)
        return x

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

        if self.is_fitted:
            return None

        print("[INFO] Computing Features: {}".format(self))
        print("[INFO] {} Training Samples".format(len(train_tags)))
        x_train = self.get_x_y(train_tags)
        self.enc.fit(x_train, None)
        self.is_fitted = True
        return 0

    def save(self, model_path):
        model_path = os.path.join(
            model_path,
            self.layer_type + "-" + self.name
        )
        check_n_make_dir(model_path)
        self.enc.save(model_path)
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))
        for p in self.previous:
            p.save(model_path)

    def load(self, model_path):
        self.opt = load_dict(os.path.join(model_path, "opt.json"))
        self.enc.load(model_path)

    def set_index(self, i):
        self.index = i
