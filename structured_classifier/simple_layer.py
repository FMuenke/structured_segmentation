from tqdm import tqdm
import os

from sklearn.model_selection import ParameterGrid
from utils.utils import check_n_make_dir, save_dict, load_dict

from utils.image_processing_operations import *

LIST_OF_OPERATIONS = [
    Resize,
    Invert,
    MorphologicalClosing,
    EdgeDetector,
    FrangiFilter,
    Threshold,
    MorphologicalOpening,
    MorphologicalClosing,
    ThresholdPercentile,
    MorphologicalDilatation,
    MorphologicalErosion,
    Blurring,
    TopClippingPercentile,
    CannyEdgeDetector,
    LocalNormalization,
    RemoveSmallObjects,
]


class Pipeline:
    def __init__(self, config, recall_importance=1.0):
        self.stats = {"TP": 0, "FP": 0, "FN": 0}
        self.beta = recall_importance
        self.config = config
        self.operations = []
        for layer, parameter in config:
            option_valid = False
            for Operation in LIST_OF_OPERATIONS:
                if layer == Operation.key:
                    self.operations.append(Operation(parameter))
                    option_valid = True
                    break
            if not option_valid:
                raise Exception("UNKNOWN OPTION: {}".format(layer))

    def inference(self, x_img):
        if np.max(x_img) <= 1 and np.min(x_img) >= 0:
            pass
        else:
            x_img = x_img / 255
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
        self.stats["FP"] += np.sum(np.logical_and(y_img != 1, p_img == 1))
        self.stats["FN"] += np.sum(np.logical_and(y_img == 1, p_img != 1))

    def summarize(self):
        if self.stats["TP"] + self.stats["FP"] == 0:
            pre = 0
        else:
            pre = self.stats["TP"] / (self.stats["TP"] + self.stats["FP"])
        if self.stats["TP"] + self.stats["FN"] == 0:
            rec = 0
        else:
            rec = self.stats["TP"] / (self.stats["TP"] + self.stats["FN"])

        if pre + rec == 0:
            return 0
        else:
            return (1 + self.beta**2) * pre * rec / (self.beta ** 2 * pre + rec)


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

    def __str__(self):
        return "{} {}".format(self.layer_type, self.pipeline.config)

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
        possible_configs = {Op.key: Op.list_of_parameters for Op in LIST_OF_OPERATIONS}

        for op in self.operations:
            if op not in possible_configs:
                pos_ops = [op for op in possible_configs]
                raise Exception("INVALID OPERATION OPTION. CHOSE: {}".format(pos_ops))
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


