import argparse
import os
import numpy as np

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
from structured_classifier.hyperparameter_optimizer import HyperParameterOptimizer

from utils import parameter_grid as pg

from utils.utils import save_dict


def run_training(df, mf, number_of_tags):
    color_coding = {
        "crack": [[255, 255, 255], [255, 0, 0]],
    }

    randomized_split = True
    train_test_ratio = 0.20

    x = InputLayer("input", features_to_use="gray-color", initial_down_scale=1)
    x = SimpleLayer(x, "SIMPLE", operations=["blurring", "edge", "threshold_percentile", "remove_small_objects"])
    model = Model(graph=x)

    d_set = SegmentationDataSet(df, color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set, percentage=train_test_ratio, random=randomized_split)

    if number_of_tags != 0:
        print("Number of training and validation samples - {}/{}".format(len(train_set), len(validation_set)))
        train_set = np.random.choice(train_set, number_of_tags, replace=False)
        number_of_tags = np.min([number_of_tags, len(validation_set)-1])
        validation_set = np.random.choice(validation_set, number_of_tags, replace=False)
        print("Number of training images reduced! - {}/{} -".format(len(train_set), len(validation_set)))

    model.fit(train_set, validation_set)
    model.save(mf)
    save_dict(color_coding, os.path.join(mf, "color_coding.json"))


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder

    number_of_images = [0, 1, 2, 5, 10, 25, 50, 100]

    for n in number_of_images:
        for i in range(5):
            if os.path.isdir(os.path.join(mf, "-{}-RUN-{}".format(n, i))):
                continue
            run_training(df, os.path.join(mf, "-{}-RUN-{}".format(n, i)), n)


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
