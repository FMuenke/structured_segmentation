import os
from utils.utils import check_n_make_dir, load_dict

from structured_classifier.decision_layer import DecisionLayer
from structured_classifier.input_layer import InputLayer
from structured_classifier.global_context_layer import GlobalContextLayer


class Model:
    def __init__(self, graph):
        self.graph = graph

        self.description = dict()

    def fit(self, train_tags, validation_tags):
        print("===============================")
        print("=====Begin Model Training======")
        print("===============================")
        self.graph.fit(train_tags, validation_tags, reduction_factor=0)

    def save(self, model_path):
        check_n_make_dir(model_path)
        check_n_make_dir(os.path.join(model_path, "graph"))
        self.graph.save(os.path.join(model_path, "graph"))

    def load(self, model_path):
        if os.path.isdir(model_path):
            if os.path.isdir(os.path.join(model_path, "graph")):
                graph_start = os.listdir(os.path.join(model_path, "graph"))[0]
                layer = self.load_layer(os.path.join(model_path, "graph", graph_start))
                layer.load(os.path.join(model_path, "graph", graph_start))
                self.graph = layer

        print("Model was loaded:")
        print(self.graph)

    def predict(self, data):
        return self.graph.predict(data)

    def load_layer(self, model_folder):
        opt = load_dict(os.path.join(model_folder, "opt.json"))
        if "layer_type" not in opt:
            raise ValueError("No LayerType Option is defined!")

        if opt["layer_type"] == "DECISION_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = DecisionLayer(prev_layer, opt["name"], opt["kernel"], opt["kernel_shape"], opt["down_scale"])
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        if opt["layer_type"] == "INPUT_LAYER":
            layer = InputLayer(opt["name"], opt["features_to_use"], height=opt["height"], width=opt["width"])
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        if opt["layer_type"] == "GLOBAL_CONTEXT_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = GlobalContextLayer(prev_layer, opt["name"], opt["down_scale"])
            layer.set_index(int(opt["index"]))
            return layer

        print(opt)
        raise ValueError("Layer not recognised!")

    def load_previous_layers(self, model_folder):
        p_layer = []
        for path in os.listdir(model_folder):
            prev_path = os.path.join(model_folder, path)
            if os.path.isdir(prev_path):
                layer = self.load_layer(prev_path)
                p_layer.append(layer)

        p_layer_sorted = [0] * len(p_layer)
        for layer in p_layer:
            p_layer_sorted[int(layer.index)] = layer
        return p_layer


