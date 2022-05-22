from tqdm import tqdm
import os
import numpy as np

from utils.utils import check_n_make_dir, save_dict, load_dict
from structured_classifier.layer_operations import resize
from structured_classifier.conventional_image_processing_pipeline.grid_search import GridSearchOptimizer
from structured_classifier.conventional_image_processing_pipeline.random_search import RandomSearchOptimizer
from structured_classifier.conventional_image_processing_pipeline.genetic_algorithm import GeneticAlgorithmOptimizer
from structured_classifier.conventional_image_processing_pipeline.pipeline import Pipeline
from structured_classifier.conventional_image_processing_pipeline.image_processing_operations import LIST_OF_OPERATIONS


def check_if_operations_are_known(operations):
    possible_options = [op.key for op in LIST_OF_OPERATIONS]
    for op in operations:
        if type(op) is list:
            check_if_operations_are_known(op)
        elif op not in possible_options:
            raise Exception("UNKNOWN OPERATION! Try: {}".format(possible_options))


class CIPPLayer:
    layer_type = "CIPP_LAYER"

    def __init__(self, INPUTS, name, operations, selected_layer, optimizer="grid_search", use_multiprocessing=True):
        self.name = str(name)
        if type(INPUTS) is not list:
            INPUTS = [INPUTS]
        self.previous = INPUTS

        self.index = 0

        for i, p in enumerate(self.previous):
            p.set_index(i)
        self.is_fitted = False

        self.operations = operations
        self.optimizer = optimizer

        self.config = None
        self.pipeline = None
        if operations is not None:
            if "fill_contours" in operations:
                print("ALERT: Option: fill_contour does not support multiprocessing during Training")
                use_multiprocessing = False
            check_if_operations_are_known(operations)
        self.use_multi_processing = use_multiprocessing

        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
            "selected_layer": selected_layer,
        }

    def __str__(self):
        return "{} TARGET:{} - {}".format(self.layer_type, self.opt["selected_layer"], self.pipeline.config)

    def inference(self, x_input, interpolation="nearest"):
        x_img, x_pass = self.get_features(x_input)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
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

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

        if self.is_fitted:
            return None

        print("Fitting {} with {}".format(self.layer_type, self.operations))
        if self.optimizer == "grid_search":
            optimizer = GridSearchOptimizer(
                operations=self.operations,
                selected_layer=self.opt["selected_layer"],
                use_multi_processing=self.use_multi_processing
            )
        elif self.optimizer == "random_search":
            optimizer = RandomSearchOptimizer(
                operations=self.operations,
                selected_layer=self.opt["selected_layer"],
                use_multi_processing=self.use_multi_processing
            )
        elif self.optimizer == "genetic_algorithm":
            optimizer = GeneticAlgorithmOptimizer(
                operations=self.operations,
                selected_layer=self.opt["selected_layer"],
                use_multi_processing=self.use_multi_processing,
            )
        else:
            raise Exception("Optimizer Choice {} unknown".format(self.optimizer))

        for t in tqdm(train_tags):
            x_img = t.load_x()
            x_img, _ = self.get_features(x_img)
            h_img, w_img = x_img.shape[:2]
            y_img = t.load_y([h_img, w_img])
            optimizer.step(x_img, y_img)

        # for t in validation_tags:
        #     x_img = t.load_x()
        #     x_img, _ = self.get_features(x_img)
        #     h_img, w_img = x_img.shape[:2]
        #     y_img = t.load_y([h_img, w_img])
        #     optimizer.step_validation(x_img, y_img)

        best_pipeline, best_score = optimizer.summarize()

        print("BestScore: {} with config {} for channel: {}".format(
            best_score, best_pipeline.config, best_pipeline.selected_layer))
        self.config = best_pipeline.config
        self.opt["selected_layer"] = int(best_pipeline.selected_layer)
        self.pipeline = Pipeline(self.config, self.opt["selected_layer"])

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
        self.pipeline = Pipeline(self.config, selected_layer=self.opt["selected_layer"])

    def set_index(self, i):
        self.index = i


