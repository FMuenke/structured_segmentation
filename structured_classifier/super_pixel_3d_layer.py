import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.util import img_as_float


from structured_classifier.classifier_handler import ClassifierHandler
from structured_classifier.layer_operations import resize
from utils.utils import check_n_make_dir, save_dict


class SuperPixel3DLayer:
    layer_type = "SUPER_PIXEL_3D_LAYER"

    def __init__(self,
                 INPUTS,
                 name,
                 super_pixel_method,
                 option,
                 time_range,
                 clf="b_rf",
                 clf_options=None,
                 param_grid=None,
                 data_reduction=0):

        self.name = str(name)
        if type(INPUTS) is not list:
            INPUTS = [INPUTS]
        self.previous = INPUTS

        self.index = 0

        self.time_range = []
        m = int(time_range / 2)
        for i in range(time_range):
            self.time_range.append(int(i - m))

        for i, p in enumerate(self.previous):
            p.set_index(i)

        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
            "super_pixel_method": super_pixel_method,
            "option": option,
            "time_range": time_range,
        }
        if clf_options is None:
            clf_options = {"type": clf}
        else:
            clf_options["type"] = clf
        self.clf = ClassifierHandler(opt=clf_options)
        self.clf.new_classifier()
        self.param_grid = param_grid
        self.data_reduction = data_reduction

    def __str__(self):
        return "{} - {} - {} - TR: {} - SP: {} - {}".format(
            self.layer_type,
            self.name,
            self.clf,
            self.time_range,
            self.opt["super_pixel_method"],
            self.opt["option"])

    def get_features_for_segments(self, tensor, segments):
        if len(tensor.shape) < 3:
            tensor = np.expand_dims(tensor, axis=2)
        h, w, num_f = tensor.shape
        x_img = []
        segments = resize(segments, width=w, height=h)

        f_tensor = np.reshape(tensor, (w * h, num_f))
        f_segments = np.reshape(segments, (w * h))

        for u in np.unique(segments):
            ids = np.where(f_segments == u)[0]
            u_val = f_tensor[ids, :]
            x_seg = np.array([
                np.mean(u_val, axis=0),
                np.std(u_val, axis=0),
                np.sum(u_val, axis=0),
                np.percentile(u_val, 0.75, axis=0),
                np.percentile(u_val, 0.25, axis=0),
            ])
            x_seg = np.reshape(x_seg, (1, -1))
            x_img.append(x_seg)
        x_img = np.concatenate(x_img, axis=0)
        return x_img

    def get_y_for_segments(self, y_img, segments):
        y = []
        segments = resize(segments, width=y_img.shape[1], height=y_img.shape[0])
        for u in np.unique(segments):
            counts = np.bincount(y_img[segments == u].astype(np.int))
            y_seg = np.argmax(counts)
            y.append(y_seg)
        return np.array(y)

    def map_segments(self, segments, y_pred):
        label_map = np.zeros((segments.shape[0], segments.shape[1], 1))
        for i, u in enumerate(np.unique(segments)):
            label_map[segments == u] = y_pred[i]
        return label_map

    def generate_segments(self, x_input):
        img = img_as_float(x_input[::2, ::2])
        if "felzenszwalb" in self.opt["super_pixel_method"]:
            segments = felzenszwalb(img, scale=self.opt["option"], sigma=0.5, min_size=50)
        elif "slic" in self.opt["super_pixel_method"]:
            segments = slic(img, n_segments=self.opt["option"], compactness=10, sigma=1, start_label=1)
        elif "quickshift" in self.opt["super_pixel_method"]:
            segments = quickshift(img, kernel_size=self.opt["option"], max_dist=6, ratio=0.5)
        elif "watershed" in self.opt["super_pixel_method"]:
            gradient = sobel(rgb2gray(img))
            segments = watershed(gradient, markers=self.opt["option"], compactness=0.001)
        else:
            raise ValueError("SuperPixel-Option: {} unknown!".format(self.opt["super_pixel_method"]))
        return segments

    def inference(self, tag, interpolation="nearest"):
        x_input = tag.load_x()
        segments = self.generate_segments(x_input)
        x_img, x_pass = self.get_features(tag)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
        x_img = self.get_features_for_segments(x_img, segments)
        y_pred = self.clf.predict(x_img)
        segments = resize(segments, width=x_width, height=x_height, interpolation="nearest")
        y_img = self.map_segments(segments, y_pred)
        x_img_pass = resize(x_pass, width=o_width, height=o_height, interpolation="nearest")
        y_img = resize(y_img, width=o_width, height=o_height, interpolation=interpolation)

        if len(x_img_pass.shape) < 3:
            x_img_pass = np.expand_dims(x_img_pass, axis=2)
        if len(y_img.shape) < 3:
            y_img = np.expand_dims(y_img, axis=2)
        y_img = np.concatenate([x_img_pass, y_img], axis=2)
        return y_img

    def predict(self, tag):
        x_input = tag.load_x()
        segments = self.generate_segments(x_input)
        x_img, x_pass = self.get_features(tag)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
        x_img = self.get_features_for_segments(x_img, segments)
        y_img = self.clf.predict(x_img)
        segments = resize(segments, width=x_width, height=x_height, interpolation="nearest")
        y_img = self.map_segments(segments, y_img)
        y_img = resize(y_img, width=o_width, height=o_height, interpolation="nearest")
        return y_img

    def get_features(self, tag_3d):
        x = []
        for p in self.previous:
            for t in self.time_range:
                f_tag3d = tag_3d.get_offset_frame(t)
                x_p = p.inference(f_tag3d, interpolation="linear")
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
            use_sample = True
            if reduction_factor > 1:
                if not np.random.randint(0, reduction_factor):
                    use_sample = False

            if use_sample:
                x_img = t.load_x()
                segments = self.generate_segments(x_img)
                x_img, _ = self.get_features(t)
                h_img, w_img = x_img.shape[:2]
                x_img = self.get_features_for_segments(x_img, segments)
                y_img = t.load_y([h_img, w_img])
                y_img = self.get_y_for_segments(y_img, segments)
                assert x_img.shape[0] == y_img.shape[0], "Not equal dimensions. [{} - {}]".format(
                    x_img.shape, y_img.shape)

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
