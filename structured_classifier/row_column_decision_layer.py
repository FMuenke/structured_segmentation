import numpy as np
import cv2
import os
from tqdm import tqdm
from structured_classifier.classifier_handler import ClassifierHandler
from structured_classifier.layer_operations import resize
from utils.utils import check_n_make_dir, save_dict


class RowColumnDecisionLayer:
    layer_type = "DECISION_LAYER"

    def __init__(self,
                 INPUTS,
                 name,
                 down_scale=0,
                 clf="b_rf",
                 n_estimators=200,
                 param_grid=None):

        self.name = str(name)
        if type(INPUTS) is not list:
            INPUTS = [INPUTS]
        self.previous = INPUTS

        self.index = 0

        for i, p in enumerate(self.previous):
            p.set_index(i)
        self.max_num_samples = 500000

        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
            "kernel": kernel,
            "down_scale": down_scale
        }

        self.down_scale = down_scale

        self.clf_row = ClassifierHandler(opt={"type": clf, "n_estimators": n_estimators})
        self.clf_col = ClassifierHandler(opt={"type": clf, "n_estimators": n_estimators})
        self.clf_row.new_classifier()
        self.clf_col.new_classifier()
        self.param_grid = param_grid

    def __str__(self):
        s = ""
        s += "{} - {} - {}\n".format(self.layer_type, self.name, self.clf_row)
        s += "---------------------------\n"
        for p in self.previous:
            s += "--> {}\n".format(p)
        return s

    def inference(self, x_input, interpolation="nearest"):
        o_height, o_width = x_input.shape[:2]
        x_img = self.get_features(x_input)
        x_height, x_width = x_img.shape[:2]

        x_img_row = self.format_to_axis_view(np.copy(x_img), axis=1)
        x_img_row = np.reshape(x_img_row, (x_height, -1))

        x_img_col = self.format_to_axis_view(np.copy(x_img), axis=0)
        x_img_col = np.reshape(x_img_col, (x_width, -1))

        y_row = self.clf_row.predict(x_img_row)
        y_col = self.clf_col.predict(x_img_col)

        y_row = np.repeat(y_row, axis=1)
        y_col = np.repeat(y_col, axis=0)

        n_classes = probs.shape[1]
        y_img = []
        for i in range(n_classes):
            y_i_img = np.reshape(probs[:, i], (x_height, x_width, 1))
            y_img.append(y_i_img)
        y_img = np.concatenate(y_img, axis=2)

        y_img = resize(y_img, width=o_width, height=o_height, interpolation=interpolation)

        if len(x_input.shape) < 3:
            x_input = np.expand_dims(x_input, axis=2)
        if len(y_img.shape) < 3:
            y_img = np.expand_dims(y_img, axis=2)
        y_img = np.concatenate([x_input, y_img], axis=2)

        return y_img

    def get_features(self, x_input):
        o_height, o_width = x_input.shape[:2]
        x = []
        for p in self.previous:
            x_p = p.inference(x_input, interpolation="cubic")
            if len(x_p.shape) < 3:
                x_p = np.expand_dims(x_p, axis=2)
            x.append(x_p)
        x = np.concatenate(x, axis=2)
        new_height = int(o_height / 2**self.down_scale)
        new_width = int(o_width / 2 ** self.down_scale)
        x = cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        if len(x.shape) < 3:
            x = np.expand_dims(x, axis=2)
        return x

    def format_to_axis_view(self, x_img, axis):
        return np.sum(x_img, axis=axis)

    def get_x_y(self, tag_set, axis):
        x = []
        y = []

        for t in tqdm(tag_set):
            x_img = t.load_x()
            x_img = self.get_features(x_img)
            h_img, w_img = x_img.shape[:2]
            y_img = t.load_y([h_img, w_img])

            if axis == "row":
                x_img = self.format_to_axis_view(x_img, axis=1)
                y_img = np.max(y_img, axis=1)
                x_img = np.reshape(x_img, (h_img, -1))
                y_img = np.reshape(y_img, h_img)
            else:
                x_img = self.format_to_axis_view(x_img, axis=0)
                y_img = np.max(y_img, axis=0)
                x_img = np.reshape(x_img, (w_img, -1))
                y_img = np.reshape(y_img, w_img)

            x.append(x_img)
            y.append(y_img)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        return x, y

    def _fit(self, train_tags, validation_tags, clf, axis):
        print("Collecting Features for Stage: {} - AXIS:".format(self, axis))
        x_train, y_train = self.get_x_y(train_tags, axis)
        x_val, y_val = self.get_x_y(validation_tags, axis)

        n_samples_train, n_features = x_train.shape

        if n_samples_train > self.max_num_samples:
            data_reduction_factor = int(n_samples_train / self.max_num_samples)
        else:
            data_reduction_factor = None

        if data_reduction_factor is not None:
            x_train, y_train = x_train[::data_reduction_factor, :], y_train[::data_reduction_factor]

        n_samples_train, n_features = x_train.shape
        n_samples_val = x_val.shape[0]
        print("DataSet has {} Samples (Train: {}/ Validation: {}) with {} features.".format(
            n_samples_train + n_samples_val, n_samples_train, n_samples_val, n_features
        ))
        if self.param_grid is not None:
            self.clf_row.fit_inc_hyper_parameter(x_train, y_train, self.param_grid, n_iter=300)
        else:
            clf.fit(x_train, y_train)
        clf.evaluate(x_val, y_val)

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

        self._fit(train_tags, validation_tags, self.clf_row, axis="row")
        self._fit(train_tags, validation_tags, self.clf_col, axis="col")

    def save(self, model_path):
        model_path = os.path.join(model_path, self.layer_type + "-" + self.name)
        check_n_make_dir(model_path)
        self.clf_row.save(model_path, "clf_row")
        self.clf_col.save(model_path, "clf_col")
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))
        for p in self.previous:
            p.save(model_path)

    def load(self, model_path):
        self.clf_row.load(model_path, "clf_row")
        self.clf_col.load(model_path, "clf_col")

    def set_index(self, i):
        self.index = i
