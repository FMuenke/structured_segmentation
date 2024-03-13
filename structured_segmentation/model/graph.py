import os
import cv2
from tqdm import tqdm
from time import time
from structured_segmentation.data_structure import ModelStatistics
from structured_segmentation.utils.utils import check_n_make_dir, load_dict
from structured_segmentation.utils.segmention_mask import convert_cls_to_color, side_by_side

from structured_segmentation.layers import StructuredClassifierLayer
from structured_segmentation.layers import StructuredEncoderLayer
from structured_segmentation.layers import InputLayer
from structured_segmentation.layers import NormalizationLayer
from structured_segmentation.layers import BottleNeckLayer
from structured_segmentation.layers import VotingLayer
from structured_segmentation.layers import SuperPixelLayer
from structured_segmentation.layers import ObjectSelectionLayer


class Graph:
    def __init__(self, layer_stack):
        self.graph = layer_stack
        self.description = dict()

    def fit(self, train_tags, validation_tags=None):
        print("===============================")
        print("=====Begin Model Training======")
        print("===============================")
        self.graph.fit(train_tags, validation_tags)

    def evaluate(self, tags, color_coding, results_folder, is_unsupervised=False, plot_results=True):
        print("[INFO] Begin Model Evaluation")
        if plot_results:
            res_folder = os.path.join(results_folder, "segmentations")
            check_n_make_dir(res_folder, clean=True)
            vis_folder = os.path.join(results_folder, "overlays")
            check_n_make_dir(vis_folder, clean=True)
            sbs_folder = os.path.join(results_folder, "side_by_side")
            check_n_make_dir(sbs_folder, clean=True)

        t0 = time()
        sh = ModelStatistics(color_coding)
        for tag in tqdm(tags):
            image = tag.load_x()
            cls_map = self.predict(image)
            color_map = convert_cls_to_color(cls_map, color_coding, unsupervised=is_unsupervised)

            if not is_unsupervised:
                tag.eval(color_map, sh)

            if plot_results:
                tag.write_result(res_folder, color_map)
                tag.visualize_result(vis_folder, color_map)
                gtr_map = tag.load_y_as_color_map(color_map.shape)
                cv2.imwrite(
                    os.path.join(sbs_folder, "{}.png".format(tag.id)),
                    side_by_side(image, gtr_map, color_map)
                )

        with open(os.path.join(results_folder, "time_prediction.txt"), "w") as f:
            f.write("[INFO] done in %0.3fs" % (time() - t0))
        sh.eval()
        sh.show()
        sh.write_report(os.path.join(results_folder, "report.txt"))

    def save(self, model_path):
        check_n_make_dir(model_path)
        check_n_make_dir(os.path.join(model_path, "graph"))
        self.graph.save(os.path.join(model_path, "graph"))
        print("Model was saved to: {}".format(model_path))

    def load(self, model_path):
        print("Loading Model from: {}".format(model_path))
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

        if opt["layer_type"] == "PIXEL_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = StructuredClassifierLayer(
                prev_layer,
                opt["name"],
                opt["kernel"],
                opt["strides"],
                opt["kernel_shape"],
                opt["down_scale"]
            )
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer
        
        if opt["layer_type"] == "ENCODER_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = StructuredEncoderLayer(
                prev_layer,
                opt["name"],
                opt["kernel"],
                opt["strides"],
                opt["kernel_shape"],
                opt["down_scale"]
            )
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        if opt["layer_type"] == "INPUT_LAYER":
            layer = InputLayer(
                opt["name"],
                opt["features_to_use"],
                height=opt["height"],
                width=opt["width"],
                initial_down_scale=opt["down_scale"]
            )
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        if opt["layer_type"] == "NORMALIZATION_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = NormalizationLayer(
                prev_layer,
                opt["name"],
                norm_option=opt["norm_option"]
            )
            layer.set_index(int(opt["index"]))
            return layer

        if opt["layer_type"] == "BOTTLE_NECK_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = BottleNeckLayer(prev_layer, opt["name"])
            layer.set_index(int(opt["index"]))
            return layer

        if opt["layer_type"] == "VOTING_Layer":
            prev_layer = self.load_previous_layers(model_folder)
            layer = VotingLayer(prev_layer, opt["name"])
            layer.set_index(int(opt["index"]))
            return layer

        if opt["layer_type"] == "SUPER_PIXEL_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = SuperPixelLayer(
                prev_layer,
                opt["name"],
                super_pixel_method=opt["super_pixel_method"],
                down_scale=opt["down_scale"],
                feature_aggregation=opt["feature_aggregation"]
            )
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        if opt["layer_type"] == "OBJECT_SELECTION_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            layer = ObjectSelectionLayer(prev_layer, opt["name"])
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        raise ValueError("Layer: {} not recognised!".format(opt["layer_type"]))

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
