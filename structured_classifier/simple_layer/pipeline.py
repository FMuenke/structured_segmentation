import numpy as np
from structured_classifier.simple_layer.image_processing_operations import LIST_OF_OPERATIONS
from structured_classifier.layer_operations import resize


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