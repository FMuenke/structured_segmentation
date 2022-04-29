import numpy as np
import copy
from structured_classifier.conventional_image_processing_pipeline.image_processing_operations import LIST_OF_OPERATIONS
from structured_classifier.layer_operations import resize

from sklearn.model_selection import ParameterGrid


class ConfigLine:
    def __init__(self):
        self.operations = []

    def __str__(self):
        return "-".join(self.operations)

    def clone(self):
        new_config_line = ConfigLine()
        new_config_line.operations = copy.deepcopy(self.operations)
        return new_config_line

    def split(self, operations):
        split = []
        for op in operations:
            new_line = self.clone()
            new_line.add(op)
            split.append(new_line)
        return split

    def add(self, op):
        self.operations.append(op)

    def build(self):
        list_of_configs = []
        possible_configs = {Op.key: Op.list_of_parameters for Op in LIST_OF_OPERATIONS}
        selected_configs = {op: possible_configs[op] for op in possible_configs if op in self.operations}
        for parameters in list(ParameterGrid(selected_configs)):
            cfg = [[op, parameters[op]] for op in self.operations]
            list_of_configs.append(cfg)
        return list_of_configs


def split_config_lines(operations):
    config_lines = [ConfigLine()]
    num_lines = 1
    for op in operations:
        if type(op) == list:
            num_lines += 1
            new_config_lines = []
            for line in config_lines:
                split_lines = line.split(op)
                new_config_lines += split_lines
            config_lines = new_config_lines
        else:
            for line in config_lines:
                line.add(op)
    return config_lines


def build_configs(operations):
    config_lines = split_config_lines(operations)
    list_of_configs = []
    for line in config_lines:
        list_of_configs += line.build()
    return list_of_configs


class Pipeline:
    def __init__(self, config, selected_layer, recall_importance=1.0):
        self.stats = {"TP": 0, "FP": 0, "FN": 0}
        self.beta = recall_importance
        self.config = config
        self.selected_layer = selected_layer
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

    def __str__(self):
        return "{}-{}".format(self.selected_layer, self.config)

    def inference(self, x_img):
        if len(x_img.shape) == 3:
            x_img = x_img[:, :, [self.selected_layer]]

        if np.max(x_img) <= 1 and np.min(x_img) >= 0:
            pass
        else:
            x_img = x_img / 255
        h, w = x_img.shape[:2]
        x_img = np.reshape(x_img, (h, w))
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