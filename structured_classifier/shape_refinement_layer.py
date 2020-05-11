import cv2
import os
import numpy as np
from tqdm import tqdm

from geometric_shapes.circle import Circle
from geometric_shapes.ellipse import Ellipse
from geometric_shapes.rectangle import Rectangle

from structured_classifier.regressor_handler import RegressorHandler
from structured_classifier.layer_operations import resize
from utils.utils import check_n_make_dir, save_dict


class ShapeRefinementLayer:
    layer_type = "SHAPE_REFINEMENT_LAYER"

    def __init__(self,
                 INPUTS,
                 name,
                 shape="circle",
                 down_scale=0,
                 data_reduction=0,
                 clf="rf",
                 clf_options=None,
                 param_grid=None):

        self.name = str(name)
        if type(INPUTS) is not list:
            INPUTS = [INPUTS]
        self.previous = INPUTS

        self.index = 0
        self.data_reduction = data_reduction

        for i, p in enumerate(self.previous):
            p.set_index(i)
        self.max_num_samples = 500000

        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
            "shape": shape,
            "down_scale": down_scale
        }

        self.shape = shape

        self.down_scale = down_scale
        if clf_options is None:
            clf_options = {"type": clf}
        else:
            clf_options["type"] = clf
        self.clf = RegressorHandler(opt=clf_options)
        self.clf.new_regressor()
        self.param_grid = param_grid

    def __str__(self):
        return "{} - {} - {} - DownScale: {}".format(self.layer_type, self.name, self.clf, self.down_scale)

    def inference(self, x_input, interpolation="nearest"):
        x_img, x_pass = self.get_features(x_input)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
        x_img = self.transform_features(x_img)
        shape_parameters = self.clf.predict(x_img)
        y_img = self.shape_parameters_to_label_map(shape_parameters[0], x_height, x_width)

        x_img_pass = resize(x_pass, width=o_width, height=o_height, interpolation="nearest")
        y_img = resize(y_img, width=o_width, height=o_height, interpolation=interpolation)

        if len(x_img_pass.shape) < 3:
            x_img_pass = np.expand_dims(x_img_pass, axis=2)
        if len(y_img.shape) < 3:
            y_img = np.expand_dims(y_img, axis=2)
        y_img = np.concatenate([x_img_pass, y_img], axis=2)
        return y_img

    def predict(self, x_input):
        x_img, x_pass = self.get_features(x_input)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
        x_img = self.transform_features(x_img)
        shape_parameters = self.clf.predict(x_img)
        y_img = self.shape_parameters_to_label_map(shape_parameters[0], x_height, x_width)
        y_img = resize(y_img, width=o_width, height=o_height, interpolation="nearest")
        return y_img

    def get_features(self, x_input):
        x = []
        for p in self.previous:
            x_p = p.inference(x_input, interpolation="cubic")
            if len(x_p.shape) < 3:
                x_p = np.expand_dims(x_p, axis=2)
            x.append(x_p)
        x = np.concatenate(x, axis=2)
        x_pass = np.copy(x)
        o_height, o_width = x.shape[:2]
        new_height = int(o_height / 2 ** self.down_scale)
        new_width = int(o_width / 2 ** self.down_scale)
        if new_width < 2:
            new_width = 2
        if new_height < 2:
            new_height = 2
        x = cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        return x, x_pass

    def get_shape(self):
        if self.shape == "circle":
            return Circle()
        if self.shape == "ellipse":
            return Ellipse()
        if self.shape == "rectangle":
            return Rectangle()
        raise ValueError("Shape: {} not known!".format(self.shape))

    def label_map_to_shape_parameters(self, label_map):
        s = self.get_shape()
        p = s.get_parameter(label_map)
        return np.reshape(p, (1, -1))

    def shape_parameters_to_label_map(self, shape_parameters, height, width):
        s = self.get_shape()
        return s.get_label_map(shape_parameters, height, width)

    def transform_features(self, x_img):
        # height, width = x_img.shape[:2]
        # x1 = np.sum(x_img, axis=0)
        # x2 = np.sum(x_img, axis=1)

        # x = np.concatenate([x1, x2])
        x = resize(x_img, width=20, height=20, interpolation="cubic")
        x = np.reshape(x, (1, -1))

        return x

    def get_x_y(self, tag_set, reduction_factor=0):
        x = []
        y = []
        for t in tqdm(tag_set):
            use_sample = True
            if reduction_factor > 1:
                if not np.random.randint(0, reduction_factor):
                    use_sample = False

            if use_sample:

                x_img = t.load_x()
                x_img, _ = self.get_features(x_img)
                h_img, w_img = x_img.shape[:2]
                y_img = t.load_y([h_img, w_img])

                x_img = self.transform_features(x_img)
                y_img = self.label_map_to_shape_parameters(y_img)

                x.append(x_img)
                y.append(y_img)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        return x, y

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

        print("Collecting Features for Stage: {}".format(self))
        print("Data is reduced by factor: {}".format(self.data_reduction))
        x_train, y_train = self.get_x_y(train_tags, reduction_factor=self.data_reduction)
        x_val, y_val = self.get_x_y(validation_tags)

        n_samples_train, n_features = x_train.shape
        n_samples_val = x_val.shape[0]
        print("DataSet has {} Samples (Train: {} / Validation: {}) with {} features.".format(
            n_samples_train + n_samples_val, n_samples_train, n_samples_val, n_features
        ))
        if self.param_grid is not None:
            self.clf.fit_inc_hyper_parameter(x_train, y_train, self.param_grid, n_iter=50, n_jobs=2)
        else:
            self.clf.fit(x_train, y_train)
        self.clf.evaluate(x_val, y_val)

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
