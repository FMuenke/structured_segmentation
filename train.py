import argparse
import os

from data_structure.segmentation_data_set import SegmentationDataSet
from structured_classifier.model import Model

from model.random_forrest import RandomStructuredRandomForrest
from model.pyramid_boosting import PyramidBoosting
from model.encoder_decoder import EncoderDecoder
from model.patch_work import PatchWork

from structured_classifier.input_layer import InputLayer
from structured_classifier.object_selection_layer import ObjectSelectionLayer
from structured_classifier.simple_layer import SimpleLayer
from structured_classifier.super_pixel_layer import SuperPixelLayer
from structured_classifier.graph_layer import GraphLayer
from structured_classifier.shape_refinement_layer import ShapeRefinementLayer
from structured_classifier.bottle_neck_layer import BottleNeckLayer
from structured_classifier.normalization_layer import NormalizationLayer
from structured_classifier.feature_extraction_layer import FeatureExtractionLayer
from structured_classifier.hyperparameter_optimizer import HyperparameterOptimizer

from utils import parameter_grid as pg

from utils.utils import save_dict


def main(args_):
    color_coding = {
        "crack": [[1, 1, 1], [255, 0, 0]],
    }

    randomized_split = True
    train_test_ratio = 0.20

    df = args_.dataset_folder
    mf = args_.model_folder

    x = InputLayer("input", features_to_use="gray-color")
    # x = NormalizationLayer(x, "NORM", norm_option="min_max_scaling")
    x = SimpleLayer(x, "SIMPLE", operations=["top_clipping_percentile", "local_normalization", "invert", "threshold_percentile"])
    # x = GraphLayer(x, "RF", kernel=(11, 11), kernel_shape="ellipse", clf="rf", clf_options={"n_estimators": 100})
    # x = SimpleLayer(x, "SIMPLE", operations=["threshold", "opening"])
    model = Model(graph=x)

    d_set = SegmentationDataSet(df, color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set, percentage=train_test_ratio, random=randomized_split)

    model.fit(train_set, validation_set)
    model.save(mf)
    save_dict(color_coding, os.path.join(mf, "color_coding.json"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with predictions",
    )
    parser.add_argument(
        "--model_folder", "-m", default="./test", help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
