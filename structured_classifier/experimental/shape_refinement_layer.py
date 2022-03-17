from copy import copy
import os
import numpy as np
from tqdm import tqdm

from geometric_shapes.geometric_shape import get_shape

from learner.regressor_handler import RegressorHandler
from learner.classifier_handler import ClassifierHandler
from structured_classifier.layer_operations import resize, dropout, augment_tag
from utils.utils import check_n_make_dir, save_dict


class ShapeRefinementLayer:
    layer_type = "SHAPE_REFINEMENT_LAYER"

    def __init__(self,
                 INPUTS,
                 name,
                 shape="ellipse",
                 global_kernel=(20, 20),
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
            "global_kernel": global_kernel
        }

        self.shape = shape

        self.global_kernel = global_kernel
        if clf_options is None:
            clf_opt = {"type": clf}
        else:
            clf_opt = copy(clf_options)
            clf_opt["type"] = clf

        if shape == "arbitrary":
            self.clf = ClassifierHandler(opt=clf_opt)
            self.clf.new_classifier()
        else:
            self.clf = RegressorHandler(opt=clf_opt)
            self.clf.new_regressor()
        self.param_grid = param_grid

    def __str__(self):
        return "{} - {} - {} - GlobalKernel: {}".format(self.layer_type, self.name, self.clf, self.global_kernel)

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
            x_p = p.inference(x_input, interpolation="linear")
            if len(x_p.shape) < 3:
                x_p = np.expand_dims(x_p, axis=2)
            x.append(x_p)
        x = np.concatenate(x, axis=2)
        x_pass = np.copy(x)
        return x, x_pass

    def label_map_to_shape_parameters(self, label_map):
        s = get_shape(self.shape)
        if self.opt["shape"] == "arbitrary":
            p = s.get_parameter(label_map, self.global_kernel)
        else:
            p = s.get_parameter(label_map)
        return np.reshape(p, (1, -1))

    def shape_parameters_to_label_map(self, shape_parameters, height, width):
        s = get_shape(self.shape)
        if self.opt["shape"] == "arbitrary":
            label_map = s.get_label_map(shape_parameters, height, width, self.global_kernel)
        else:
            label_map = s.get_label_map(shape_parameters, height, width)
        return label_map

    def transform_features(self, x_img):
        x = resize(x_img, width=self.global_kernel[0], height=self.global_kernel[1], interpolation="linear")
        x = np.reshape(x, (1, -1))
        return x

    def get_x_y(self, tag_set, reduction_factor=0, augment=0):
        x = None
        y = None
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

                if augment > 0:
                    for a in range(augment):
                        x_img_a, y_img_a = augment_tag(x_img, y_img)
                        x_img_a = self.transform_features(x_img_a)
                        y_img_a = self.label_map_to_shape_parameters(y_img_a)
                        if x is None:
                            x = x_img_a
                            y = y_img_a
                        else:
                            x = np.append(x, x_img_a, axis=0)
                            y = np.append(y, y_img_a, axis=0)

                x_img_a = self.transform_features(x_img)
                y_img_a = self.label_map_to_shape_parameters(y_img)

                if x is None:
                    x = x_img_a
                    y = y_img_a
                else:
                    x = np.append(x, x_img_a, axis=0)
                    y = np.append(y, y_img_a, axis=0)
        return x, y

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

        print("Collecting Features for Stage: {}".format(self))
        print("Data is reduced by factor: {}".format(self.data_reduction))
        x_train, y_train = self.get_x_y(train_tags, reduction_factor=self.data_reduction, augment=10)
        x_train = dropout(x_train, drop_rate=0.1)
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
