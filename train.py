import argparse
import os

from data_structure.segmentation_data_set import SegmentationDataSet
from structured_classifier.model import Model

from structured_classifier.input_layer import InputLayer
from structured_classifier import CIPPLayer
from structured_classifier import PixelLayer

from structured_classifier.augmentation import augment_data_set, Augmentations

from utils.utils import save_dict


def model_v4():
    x = InputLayer("IN", features_to_use=["RGB-color"], initial_down_scale=1)
    x = CIPPLayer(x, "SIMPLE", operations=[
        "threshold_percentile",
        "fill_contours",
        "closing",
        "remove_small_objects",
    ], selected_layer=[0, 1, 2], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)

    x = InputLayer("IN", features_to_use=["RGB-color"], initial_down_scale=1)
    x = CIPPLayer(x, "SIMPLE", operations=[
        "watershed",
        "closing",
        "opening",
        "remove_small_objects",
    ], selected_layer=[1], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)
    return model


def main(args_):
    color_coding = {
        "crack": [[255, 255, 255], [255, 0, 0]],
        # "shadow": [[1, 1, 1], [0, 100, 255]]
        # "heart": [[4, 4, 4], [0, 100, 255]]
        # "nuceli": [[255, 255, 255], [100, 100, 255]],
    }

    randomized_split = True
    train_test_ratio = 0.10

    df = args_.dataset_folder
    mf = args_.model_folder

    x = InputLayer("IN", features_to_use="RGB-color", initial_down_scale=1)
    x = CIPPLayer(x, "SIMPLE", operations=[
        "blurring",
        "invert",
        ["watershed", "threshold", "threshold_percentile", "edge"],
        "closing",
        "erode",
    ], selected_layer=[0, 1, 2], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)

    d_set = SegmentationDataSet(df, color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set, percentage=train_test_ratio, random=randomized_split)

    # augmentations = Augmentations(True, True, True)
    # train_set = augment_data_set(train_set, augmentations, multiplier=3)

    model.fit(train_set[:16], validation_set)
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
