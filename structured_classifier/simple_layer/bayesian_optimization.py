import bayes_opt
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier

from structured_classifier.simple_layer.pipeline import Pipeline, LIST_OF_OPERATIONS


class BayesianOptimizer:
    def __init__(self, operations, selected_layer, use_multi_processing):
        self.operations = operations
        self.selected_layer = selected_layer
        self.use_multi_processing = use_multi_processing
        self.multiprocessing_chunk_size = 500

        self.init_points = 5
        self.n_iter = 25
        self.acq = 'ucb'
        self.kappa = 2.576
        self.kappa_decay = 1
        self.kappa_decay_delay = 0
        self.xi = 0.0

        self.gp = GaussianProcessClassifier()

        self.bounds = self.build_configs()

    def build_configs(self):
        possible_configs = {Op.key: Op.list_of_parameters for Op in LIST_OF_OPERATIONS}
        return {op: possible_configs[op] for op in possible_configs if op in self.operations}

    def probe(self, x_img, y_img, parameters):
        operations = []
        selected_layer = 0
        for p in parameters:
            if p == "selected_layer":
                selected_layer = parameters[p]
            else:
                operations.append([p, parameters[p]])

        pipeline = Pipeline(config=operations, selected_layer=selected_layer)
        pipeline.eval(x_img, y_img)


    def step(self, x_img, y_img):
        pass