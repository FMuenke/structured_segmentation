from multiprocessing.pool import Pool
from sklearn.model_selection import ParameterGrid

from structured_classifier.simple_layer.pipeline import Pipeline
from structured_classifier.simple_layer.image_processing_operations import LIST_OF_OPERATIONS


def eval_pipeline(args):
    evaluated_pipelines = []
    for bundle in args:
        pl, x_img, y_img = bundle
        pl.eval(x_img, y_img)
        evaluated_pipelines.append(pl)
    return evaluated_pipelines


def flatten_list(t):
    flat_list = [item for sublist in t for item in sublist]
    return flat_list


class GridSearchOptimizer:
    def __init__(self, operations, selected_layer, use_multi_processing):
        self.operations = operations
        self.selected_layer = selected_layer
        self.use_multi_processing = use_multi_processing
        self.multiprocessing_chunk_size = 500

        if type(self.selected_layer) is not list:
            self.pipelines = [Pipeline(config, self.selected_layer) for config in self.build_configs()]
        else:
            self.pipelines = []
            for selected_index in self.selected_layer:
                self.pipelines += [Pipeline(config, selected_index) for config in self.build_configs()]
        print("Evaluating - {} - Configurations".format(len(self.pipelines)))

    def build_configs(self):
        list_of_configs = []
        possible_configs = {Op.key: Op.list_of_parameters for Op in LIST_OF_OPERATIONS}

        # for op in self.operations:
        #     if op not in possible_configs:
        #         pos_ops = [op for op in possible_configs]
        #         raise Exception("INVALID OPERATION OPTION {}. CHOSE: {}".format(op, pos_ops))
        selected_configs = {op: possible_configs[op] for op in possible_configs if op in self.operations}
        for parameters in list(ParameterGrid(selected_configs)):
            cfg = [[op, parameters[op]] for op in self.operations]
            list_of_configs.append(cfg)
        return list_of_configs

    def step(self, x_img, y_img):
        if self.use_multi_processing:
            tasks = [[pl, x_img, y_img] for pl in self.pipelines]
            n = self.multiprocessing_chunk_size
            tasks_bundled = [tasks[i:i + n] for i in range(0, len(tasks), n)]
            with Pool() as p:
                bundled_pipelines = p.map(eval_pipeline, tasks_bundled)
            self.pipelines = flatten_list(bundled_pipelines)

        else:
            for pl in self.pipelines:
                pl.eval(x_img, y_img)

    def summarize(self):
        best_score = 0
        best_pipeline = None
        for pl in self.pipelines:
            score = pl.summarize()
            if score >= best_score:
                best_score = score
                best_pipeline = pl
        return best_pipeline, best_score
