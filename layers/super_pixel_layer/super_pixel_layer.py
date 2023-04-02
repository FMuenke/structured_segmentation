import os
import numpy as np
from tqdm import tqdm

from layers.super_pixel_layer.super_pixel import get_features_for_segments, generate_segments, map_segments, get_y_for_segments

from learner.internal_classifier import InternalClassifier
from layers.layer_operations import resize
from utils.utils import check_n_make_dir, save_dict


class SuperPixelLayer:
    layer_type = "SUPER_PIXEL_LAYER"

    def __init__(self,
                 INPUTS,
                 name,
                 super_pixel_method="slic",
                 down_scale=0,
                 feature_aggregation="gauss",
                 clf="rf",
                 clf_options=None,
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
            "super_pixel_method": super_pixel_method,
            "down_scale": down_scale,
            "feature_aggregation": feature_aggregation,
        }
        if clf_options is None:
            clf_options = {"type": clf}
        else:
            clf_options["type"] = clf
        self.clf = InternalClassifier(opt=clf_options)
        self.clf.new()
        self.data_reduction = data_reduction

    def __str__(self):
        return "{} - {} - {} - SP: {}+{} - Downscale: {}".format(
            self.layer_type,
            self.name,
            self.clf,
            self.opt["super_pixel_method"],
            self.opt["feature_aggregation"],
            self.opt["down_scale"]
        )

    def inference(self, x_input, interpolation="nearest"):
        segments = generate_segments(x_input, self.opt)
        x_img, x_pass = self.get_features(x_input)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
        x_img = get_features_for_segments(x_img, segments, self.opt["feature_aggregation"])
        y_pred = self.clf.predict_proba(x_img)
        segments = resize(segments, width=x_width, height=x_height, interpolation="nearest")
        y_img = map_segments(segments, y_pred)
        x_img_pass = resize(x_pass, width=o_width, height=o_height, interpolation="nearest")
        y_img = resize(y_img, width=o_width, height=o_height, interpolation=interpolation)

        if len(x_img_pass.shape) < 3:
            x_img_pass = np.expand_dims(x_img_pass, axis=2)
        if len(y_img.shape) < 3:
            y_img = np.expand_dims(y_img, axis=2)
        y_img = np.concatenate([x_img_pass, y_img], axis=2)
        return y_img

    def predict(self, x_input):
        segments = generate_segments(x_input, self.opt)
        x_img, x_pass = self.get_features(x_input)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
        x_img = get_features_for_segments(x_img, segments, self.opt["feature_aggregation"])
        y_img = self.clf.predict(x_img)
        segments = resize(segments, width=x_width, height=x_height, interpolation="nearest")
        y_img = map_segments(segments, y_img)
        y_img = resize(y_img, width=o_width, height=o_height, interpolation="nearest")
        return y_img

    def get_features(self, x_input):
        x = []
        for p in self.previous:
            x_p = p.inference(x_input, interpolation="linear")
            if len(x_p.shape) < 3:
                x_p = np.expand_dims(x_p, axis=2)
            x.append(x_p)
        x = np.concatenate(x, axis=2)
        x_pass = np.copy(x)
        x = x.astype(np.float32)
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        return x, x_pass

    def get_x_y(self, tag_set, reduction_factor=0):
        x = None
        y = None
        for t in tqdm(tag_set):
            if np.random.randint(100) > 100 * reduction_factor:
                x_img = t.load_x()
                segments = generate_segments(x_img, self.opt)
                x_img, _ = self.get_features(x_img)
                h_img, w_img = x_img.shape[:2]
                x_img = get_features_for_segments(x_img, segments, self.opt["feature_aggregation"])
                y_img = t.load_y([h_img, w_img])
                y_img = get_y_for_segments(y_img, segments)
                assert x_img.shape[0] == y_img.shape[0], "Not equal dimensions. [{} - {}]".format(
                    x_img.shape, y_img.shape)

                if x is None:
                    x = x_img
                    y = y_img
                else:
                    x = np.append(x, x_img, axis=0)
                    y = np.append(y, y_img, axis=0)
        return x, y

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

        if self.clf.is_fitted():
            return None

        print("[INFO] Collecting Features for Stage: {}. {} Training Samples (DataReduction {})".format(
            self, len(train_tags), self.data_reduction))
        x_train, y_train = self.get_x_y(train_tags, reduction_factor=self.data_reduction)
        self.clf.fit(x_train, y_train)
        if validation_tags is None:
            return 0
        else:
            x_val, y_val = self.get_x_y(validation_tags)
            return self.clf.evaluate(x_val, y_val)

    def save(self, model_path):
        model_path = os.path.join(model_path, self.layer_type + "-" + self.name)
        check_n_make_dir(model_path)
        self.clf.save(model_path)
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))
        for p in self.previous:
            p.save(model_path)

    def load(self, model_path):
        self.clf.load(model_path)

    def set_index(self, i):
        self.index = i
