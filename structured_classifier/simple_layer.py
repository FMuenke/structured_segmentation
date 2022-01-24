from tqdm import tqdm
import numpy as np
import cv2
import os
from multiprocessing import Pool

from structured_classifier.layer_operations import normalize, resize

from sklearn.model_selection import ParameterGrid
from utils.utils import check_n_make_dir, save_dict, load_dict


class EdgeDetector:
    list_of_parameters = [
        None, 3, 5, 7, 9, 11, 13, 15
    ]

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        sobel_x = cv2.Sobel(x_img, cv2.CV_64F, 1, 0, ksize=self.parameter)
        sobel_y = cv2.Sobel(x_img, cv2.CV_64F, 0, 1, ksize=self.parameter)
        return np.sqrt(sobel_x**2 + sobel_y**2)


class Threshold:
    list_of_parameters = [
        None, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9
    ]

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = normalize(x_img)
        x_img[x_img < self.parameter] = 0
        x_img[x_img >= self.parameter] = 1
        return x_img


class MorphologicalOpening:
    list_of_parameters = [
        None, 3, 5, 7, 9, 11, 13, 15
    ]

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        kernel = np.ones((self.parameter, self.parameter), np.uint8)
        return cv2.morphologyEx(x_img, cv2.MORPH_OPEN, kernel)


class MorphologicalClosing:
    list_of_parameters = [
        None, 3, 5, 7, 9, 11, 13, 15
    ]

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        kernel = np.ones((self.parameter, self.parameter), np.uint8)
        return cv2.morphologyEx(x_img, cv2.MORPH_CLOSE, kernel)


class Invert:
    list_of_parameters = [
        -1, 1
    ]

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter == -1:
            stamp = np.ones(x_img.shape) * np.max(x_img)
            return stamp - x_img
        else:
            return x_img


class Resize:
    list_of_parameters = [
        None, 32, 64, 128, 256, 512
    ]

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        return resize(x_img, width=self.parameter, height=self.parameter)


class Pipeline:
    def __init__(self, config):
        self.stats = {"TP": 0, "FP": 0, "FN": 0}
        self.config = config
        self.operations = []
        for layer, parameter in config:
            if layer == "edge":
                self.operations.append(EdgeDetector(parameter))
            elif layer == "threshold":
                self.operations.append(Threshold(parameter))
            elif layer == "opening":
                self.operations.append(MorphologicalOpening(parameter))
            elif layer == "closing":
                self.operations.append(MorphologicalClosing(parameter))
            elif layer == "invert":
                self.operations.append(Invert(parameter))
            elif layer == "resize":
                self.operations.append(Resize(parameter))
            else:
                raise ValueError("OPERATION IS NOT KNOWN: {}".format(layer))

    def inference(self, x_img):
        if len(x_img.shape) == 3:
            x_img = x_img[:, :, -1]
        for op in self.operations:
            x_img = op.inference(x_img)
        return x_img

    def eval(self, x_img, y_img):
        p_img = self.inference(x_img)
        h_img, w_img = p_img.shape[:2]
        y_img = resize(y_img, w_img, h_img)
        y_img = np.reshape(y_img, h_img * w_img)
        p_img = np.reshape(p_img, h_img * w_img)

        self.stats["TP"] += np.sum(np.logical_and(y_img == 1, p_img == 1))
        self.stats["FP"] += np.sum(np.logical_and(y_img == 0, p_img == 1))
        self.stats["FN"] += np.sum(np.logical_and(y_img == 1, p_img == 0))

    def summarize(self):
        pre = self.stats["TP"] / (self.stats["TP"] + self.stats["FP"] + 1e-5)
        rec = self.stats["TP"] / (self.stats["TP"] + self.stats["FN"] + 1e-5)
        return 2 * pre * rec / (pre + rec + 1e-5)


class SimpleLayer:
    layer_type = "SIMPLE_LAYER"

    def __init__(self, INPUTS, name, operations):
        self.name = str(name)
        if type(INPUTS) is not list:
            INPUTS = [INPUTS]
        self.previous = INPUTS

        self.index = 0

        for i, p in enumerate(self.previous):
            p.set_index(i)
        self.is_fitted = False

        self.operations = operations

        self.config = None
        self.pipeline = None

        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
        }

    def inference(self, x_input, interpolation="nearest"):
        x_img, x_pass = self.get_features(x_input)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
        x_img = np.reshape(x_img, (x_height * x_width, -1))
        y_img = self.pipeline.inference(x_img)
        y_img = np.reshape(y_img, (x_height, x_width, 1))

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
        y_img = self.pipeline.inference(x_img)
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

    def build_configs(self):
        list_of_configs = []
        possible_configs = {
            "invert": Invert.list_of_parameters,
            "edge": EdgeDetector.list_of_parameters,
            "threshold": Threshold.list_of_parameters,
            "closing": MorphologicalClosing.list_of_parameters,
            "opening": MorphologicalOpening.list_of_parameters,
            "resize": Resize.list_of_parameters,
        }
        selected_configs = {op: possible_configs[op] for op in possible_configs if op in self.operations}
        for parameters in list(ParameterGrid(selected_configs)):
            cfg = [[op, parameters[op]] for op in self.operations]
            list_of_configs.append(cfg)
        print("Evaluating - {} - Configurations".format(len(list_of_configs)))
        return list_of_configs

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

        if self.is_fitted:
            return None

        print("Fitting {} with {}".format(self.layer_type, self.operations))
        pipelines = [Pipeline(config) for config in self.build_configs()]

        for t in tqdm(train_tags):
            x_img = t.load_x()
            x_img, _ = self.get_features(x_img)
            h_img, w_img = x_img.shape[:2]
            y_img = t.load_y([h_img, w_img])

            for pl in pipelines:
                pl.eval(x_img, y_img)

        best_score = 0
        best_pipeline = None
        for pl in pipelines:
            score = pl.summarize()
            if score >= best_score:
                best_score = score
                best_pipeline = pl

        print("BestScore: {} with config {}".format(best_score, best_pipeline.config))
        self.config = best_pipeline.config
        self.pipeline = Pipeline(self.config)

        for t in validation_tags:
            x_img = t.load_x()
            x_img, _ = self.get_features(x_img)
            h_img, w_img = x_img.shape[:2]
            y_img = t.load_y([h_img, w_img])
            self.pipeline.eval(x_img, y_img)

        print("Validation: {}".format(self.pipeline.summarize()))

    def save(self, model_path):
        model_path = os.path.join(model_path, self.layer_type + "-" + self.name)
        check_n_make_dir(model_path)
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))
        save_dict(self.config, os.path.join(model_path, "config.json"))
        for p in self.previous:
            p.save(model_path)

    def load(self, model_path):
        self.config = load_dict(os.path.join(model_path, "config.json"))
        self.pipeline = Pipeline(self.config)

    def set_index(self, i):
        self.index = i

