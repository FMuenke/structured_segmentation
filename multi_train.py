import argparse
import os
import numpy as np

from data_structure.segmentation_data_set import SegmentationDataSet
from structured_classifier.model import Model

from structured_classifier.input_layer import InputLayer
from structured_classifier.simple_layer.simple_layer import SimpleLayer

from utils.utils import save_dict


def run_training(df, mf, number_of_tags):
    color_coding = {
        "ellipse": [[255, 255, 255], [255, 0, 0]],
    }

    randomized_split = True
    train_test_ratio = 0.20

    x = InputLayer("IN", features_to_use="gray-color", initial_down_scale=1)
    x = SimpleLayer(x, "SIMPLE", operations=[
        "blurring",
        "top_clipping_percentile",
        "negative_closing",
        "threshold",
        "remove_small_objects",
    ], selected_layer=[0], use_multiprocessing=True)
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

    if not os.path.isdir(mf):
        os.mkdir(mf)

    number_of_images = [1, 2, 5, 10, 25]
    iterations = 20

    for n in number_of_images:
        for i in range(iterations):
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
