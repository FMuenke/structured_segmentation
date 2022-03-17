import argparse
import os

from data_structure.segmentation_data_set import SegmentationDataSet
from structured_classifier.model import Model

from structured_classifier.input_layer import InputLayer
from structured_classifier.simple_layer.simple_layer import SimpleLayer

from utils.utils import save_dict


def main(args_):
    color_coding = {
        "crack": [[255, 255, 255], [255, 0, 0]],
        # "shadow": [[1, 1, 1], [0, 100, 255]]
        # "heart": [[4, 4, 4], [0, 100, 255]]
        # "nuceli": [[255, 255, 255], [100, 100, 255]],
    }

    randomized_split = True
    train_test_ratio = 0.90

    df = args_.dataset_folder
    mf = args_.model_folder

    x = InputLayer("IN", features_to_use=["gray-color"], initial_down_scale=1)
    # x = NormalizationLayer(x, "NORM")
    x = SimpleLayer(x, "SIMPLE", operations=[
        "blurring",
        "top_clipping_percentile",
        "negative_closing",
        "threshold",
        "remove_small_objects",
    ], selected_layer=[0], optimizer="genetic_algorithm", use_multiprocessing=True)

    model = Model(graph=x)

    d_set = SegmentationDataSet(df, color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set, percentage=train_test_ratio, random=randomized_split)

    model.fit(train_set[:10], validation_set)
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
