import os
from utils.utils import check_n_make_dir, load_dict

from structured_classifier.decision_layer import DecisionLayer
from structured_classifier.decision_3d_layer import Decision3DLayer
from structured_classifier.input_layer import InputLayer
from structured_classifier.input_3d_layer import Input3DLayer
from structured_classifier.global_context_layer import GlobalContextLayer
from structured_classifier.normalization_layer import NormalizationLayer
from structured_classifier.shape_refinement_layer import ShapeRefinementLayer
from structured_classifier.bottle_neck_layer import BottleNeckLayer


class Model:
    def __init__(self, graph):
        self.graph = graph

        self.description = dict()

    def fit(self, train_tags, validation_tags):
        print("===============================")
        print("=====Begin Model Training======")
        print("===============================")
        self.graph.fit(train_tags, validation_tags)

    def save(self, model_path):
        check_n_make_dir(model_path)
        check_n_make_dir(os.path.join(model_path, "graph"))
        self.graph.save(os.path.join(model_path, "graph"))
        print("Model was saved to: {}".format(model_path))

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

        if opt["layer_type"] == "DECISION3D_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = Decision3DLayer(prev_layer, opt["name"], opt["kernel"], opt["kernel_shape"], opt["down_scale"])
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        if opt["layer_type"] == "INPUT_LAYER":
            layer = InputLayer(opt["name"], opt["features_to_use"], height=opt["height"], width=opt["width"],
                               initial_down_scale=opt["down_scale"])
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        if opt["layer_type"] == "INPUT3D_LAYER":
            layer = Input3DLayer(opt["name"], opt["features_to_use"], height=opt["height"], width=opt["width"],
                               initial_down_scale=opt["down_scale"])
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        if opt["layer_type"] == "GLOBAL_CONTEXT_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = GlobalContextLayer(prev_layer, opt["name"], opt["down_scale"])
            layer.set_index(int(opt["index"]))
            return layer

        if opt["layer_type"] == "NORMALIZATION_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = NormalizationLayer(prev_layer, opt["name"], norm_option=opt["norm_option"])
            layer.set_index(int(opt["index"]))
            return layer

        if opt["layer_type"] == "SHAPE_REFINEMENT_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = ShapeRefinementLayer(prev_layer, opt["name"], shape=opt["shape"], global_kernel=opt["global_kernel"])
            layer.load(model_folder)
            layer.set_index(int(opt["index"]))
            return layer

        if opt["layer_type"] == "BOTTLE_NECK_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = BottleNeckLayer(prev_layer, opt["name"])
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


