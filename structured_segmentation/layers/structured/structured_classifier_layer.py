import cv2
import os
import logging
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from structured_segmentation.layers.structured.kernel import Kernel
from structured_segmentation.learner.internal_classifier import InternalClassifier
from structured_segmentation.layers.layer_operations import resize
from structured_segmentation.utils.utils import check_n_make_dir, save_dict


class StructuredClassifierLayer:
    layer_type = "PIXEL_LAYER"

    def __init__(self,
                 INPUTS,
                 name,
                 kernel=(1, 1),
                 strides=(1, 1),
                 kernel_shape="square",
                 down_scale=0,
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
        self.imbalance_threshold = 0.01
        self.imbalance_downsample = 0.9
        self.downsample_threshold = 0.4

        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
            "kernel": kernel,
            "kernel_shape": kernel_shape,
            "down_scale": down_scale,
            "strides": strides,
        }
        self.is_fitted = False

        self.down_scale = down_scale
        if clf_options is None:
            clf_options = {"type": clf}
        else:
            clf_options["type"] = clf
        self.clf = InternalClassifier(opt=clf_options)
        self.clf.new()
        self.data_reduction = data_reduction

        self.kernel = Kernel(kernel, strides, kernel_shape)

    def __str__(self):
        return "{} - {} - {} - DownScale: {} - K: {}".format(
            self.layer_type,
            self.name,
            self.clf,
            self.down_scale,
            self.opt["kernel"]
        )

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
            x_p = p.inference(x_input)
            if len(x_p.shape) < 3:
                x_p = np.expand_dims(x_p, axis=2)
            x.append(x_p)
        x = np.concatenate(x, axis=2)
        x_pass = np.copy(x)
        o_height, o_width = x.shape[:2]
        new_height = np.max([int(o_height / 2 ** self.down_scale), 2])
        new_width = np.max([int(o_width / 2 ** self.down_scale), 2])
        x = cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        x = self.kernel.get_kernel(x)
        x = x.astype(np.float32)
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        return x, x_pass

    def get_x_y(self, tag_set, reduction_factor=0):
        x = None
        y = None
        for t in tqdm(tag_set):
            if np.random.randint(100) < 100 * reduction_factor:
                continue
            x_img = t.load_x()
            x_img, _ = self.get_features(x_img)
            h_img, w_img = x_img.shape[:2]
            y_img = t.load_y([h_img, w_img])

            x_img = np.reshape(x_img, (h_img * w_img, -1))
            y_img = np.reshape(y_img, h_img * w_img)

            # Remove unlabeled samples
            x_img = x_img[y_img != -1, :]
            y_img = y_img[y_img != -1]

            if len(y_img) == 0:
                continue

            n_samples_in_sample = x_img.shape[0]

            u, c = np.unique(y_img, return_counts=True)
            p = c / n_samples_in_sample
            if len(u) == 1:
                if reduction_factor > 0 and np.min(c) > 16:
                    x_img, _, y_img, _ = train_test_split(x_img, y_img, test_size=reduction_factor)
            elif np.min(p) < self.imbalance_threshold:
                logging.info("Strong Imbalance is detected. DownSample Large Classes...")
                x_img_select, y_img_select = [], []
                for cls_id, num_cls, percentage_cls in zip(u, c, p):
                    x_img_c, y_img_c = x_img[y_img == cls_id, :], y_img[y_img == cls_id]
                    if percentage_cls > self.downsample_threshold and num_cls > 16:
                        x_img_c, _, y_img_c, _ = train_test_split(x_img_c, y_img_c, test_size=self.imbalance_downsample)
                    x_img_select.append(x_img_c)
                    y_img_select.append(y_img_c)

                x_img, y_img = np.concatenate(x_img_select, axis=0), np.concatenate(y_img_select, axis=0)
            elif reduction_factor > 0 and np.min(c) > 16:
                x_img, _, y_img, _ = train_test_split(x_img, y_img, test_size=reduction_factor, stratify=y_img)

            if x is None:
                x = x_img
                y = y_img
            else:
                x = np.append(x, x_img, axis=0)
                y = np.append(y, y_img, axis=0)
        x = x.astype(np.float32)
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        return x, y

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

        if self.is_fitted:
            return None

        print("[INFO] Computing Features: {}".format(self))
        print("[INFO] {} Training Samples (DataReduction {})".format(
            len(train_tags),
            self.data_reduction
        ))
        x_train, y_train = self.get_x_y(train_tags, reduction_factor=self.data_reduction)
        self.clf.fit(x_train, y_train)
        self.is_fitted = True
        if validation_tags is None:
            return 0
        else:
            x_val, y_val = self.get_x_y(validation_tags)
            return self.clf.evaluate(x_val, y_val)

    def save(self, model_path):
        model_path = os.path.join(
            model_path,
            self.layer_type + "-" + self.name
        )
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
