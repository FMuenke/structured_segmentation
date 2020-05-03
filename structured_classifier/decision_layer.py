import cv2
import os
import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve


from structured_classifier.classifier_handler import ClassifierHandler
from structured_classifier.layer_operations import resize
from utils.utils import check_n_make_dir, save_dict


class DecisionLayer:
    layer_type = "DECISION_LAYER"

    def __init__(self,
                 INPUTS,
                 name,
                 kernel=(1, 1),
                 kernel_shape="square",
                 down_scale=0,
                 clf="b_rf",
                 clf_options=None,
                 param_grid=None,
                 data_reduction=0):

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
            "kernel_shape": kernel_shape,
            "down_scale": down_scale
        }

        self.down_scale = down_scale
        if clf_options is None:
            clf_options = {"type": clf}
        else:
            clf_options["type"] = clf
        self.clf = ClassifierHandler(opt=clf_options)
        self.clf.new_classifier()
        self.param_grid = param_grid
        self.data_reduction = data_reduction

        k_x, k_y = kernel
        s_element = self.make_s_element(kernel, kernel_shape)
        self.look_ups = []
        if kernel is not None:
            for i in range(k_x):
                for j in range(k_y):
                    if s_element[j, i] == 1:
                        look = np.zeros((k_y, k_x, 1))
                        look[j, i, 0] = 1
                        self.look_ups.append(look)

    def make_s_element(self, kernel, kernel_shape):
        k_x, k_y = kernel
        if kernel_shape == "square":
            return np.ones((k_y, k_x))
        if kernel_shape == "ellipse":
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_y, k_x))
        if kernel_shape == "cross":
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (k_y, k_x))
        raise ValueError("Kernel-Shape option: {} not known".format(kernel_shape))

    def __str__(self):
        return "{} - {} - {} - DownScale: {}".format(self.layer_type, self.name, self.clf, self.down_scale)

    def get_kernel(self, tensor):
        if len(tensor.shape) < 3:
            tensor = np.expand_dims(tensor, axis=2)
        num_f = tensor.shape[2]
        look_up_tensors = []
        for look in self.look_ups:
            filter_block = np.repeat(look, num_f, axis=2)
            tens = convolve(tensor, filter_block)
            look_up_tensors.append(tens)
        return np.concatenate(look_up_tensors, axis=2)

    def inference(self, x_input, interpolation="nearest"):
        x_img, x_pass = self.get_features(x_input)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
        x_img = np.reshape(x_img, (x_height * x_width, -1))
        probs = self.clf.predict_proba(x_img)
        n_classes = probs.shape[1]
        y_img = []
        for i in range(n_classes):
            y_i_img = np.reshape(probs[:, i], (x_height, x_width, 1))
            y_img.append(y_i_img)
        y_img = np.concatenate(y_img, axis=2)

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
        x_img = np.reshape(x_img, (x_height * x_width, -1))
        y_img = self.clf.predict(x_img)
        y_img = np.reshape(y_img, (x_height, x_width))
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
        new_height = int(o_height / 2**self.down_scale)
        new_width = int(o_width / 2 ** self.down_scale)
        if new_width < 2:
            new_width = 2
        if new_height < 2:
            new_height = 2
        x = cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        x = self.get_kernel(x)
        return x, x_pass

    def get_x_y(self, tag_set, reduction_factor=0):
        x = []
        y = []
        n = 0
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

                x_img = np.reshape(x_img, (h_img * w_img, -1))
                y_img = np.reshape(y_img, h_img * w_img)

                n_samples_train, n_features = x_img.shape
                num_allowed_data = self.max_num_samples / len(tag_set)
                if n_samples_train > num_allowed_data:
                    data_reduction_factor = int(n_samples_train / num_allowed_data)
                else:
                    data_reduction_factor = None

                if data_reduction_factor is not None:
                    x_img, y_img = x_img[::data_reduction_factor, :], y_img[::data_reduction_factor]

                n += x_img.shape[0]
                # print(n)

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
