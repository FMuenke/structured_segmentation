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
from structured_classifier.pixel_layer import PixelLayer
from structured_classifier.bottle_neck_layer import BottleNeckLayer
from structured_classifier.normalization_layer import NormalizationLayer
from structured_classifier.hyperparameter_optimizer import HyperParameterOptimizer

from utils import parameter_grid as pg

from utils.utils import save_dict


def main(args_):
    color_coding = {
        "crack": [[255, 255, 255], [255, 0, 0]],
        # "shadow": [[1, 1, 1], [0, 100, 255]]
    }

    randomized_split = True
    train_test_ratio = 0.80

    df = args_.dataset_folder
    mf = args_.model_folder

    x = InputLayer("IN", features_to_use="gray-color", initial_down_scale=1)
    x = SimpleLayer(x, "SIMPLE",
                    operations=["blurring", "edge", "threshold_percentile", "remove_small_objects"],
                    selected_layer=[0], use_multiprocessing=True)
    # x = PixelLayer(x, "px", (3, 3), (3, 3))
    # x = SuperPixelLayer(x, "SP", down_scale=2, feature_aggregation="hist16", clf="rf")
    # x = ObjectSelectionLayer(x, "SELECT")
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
        "--model_folder",
        "-model",
        default="./test",
        help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
