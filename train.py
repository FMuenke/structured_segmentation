import argparse
import os

from data_structure.data_set import DataSet
from structured_classifier.model import Model

from structured_classifier.decision_layer import DecisionLayer
from structured_classifier.input_layer import InputLayer
from structured_classifier.global_context_layer import GlobalContextLayer
from structured_classifier.normalization_layer import NormalizationLayer

from utils import parameter_grid as pg

from utils.utils import save_dict


def main(args_):
    color_coding = {
        # "man_hole": [[1, 1, 1], [0, 255, 0]],
        # "crack": [[3, 3, 3], [255, 255, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        "shadow": [[1, 1, 1], [255, 0, 0]],
        # "filled_crack": [[2, 2, 2], [0, 255, 0]],
    }

    randomized_split = False
    train_test_ratio = 0.20

    df = args_.dataset_folder
    mf = args_.model_folder

    x1 = InputLayer("input_1", ["hsv-color"], initial_down_scale=2)
    x1 = DecisionLayer(INPUTS=x1, name="decision_0", kernel=(5, 5), kernel_shape="ellipse", down_scale=1,
                       n_estimators=25, data_reduction=3)
    x1 = DecisionLayer(INPUTS=x1, name="decision_1", kernel=(5, 5), kernel_shape="ellipse", down_scale=2,
                       n_estimators=25, data_reduction=3)
    x1 = DecisionLayer(INPUTS=x1, name="decision_2", kernel=(5, 5), kernel_shape="ellipse", down_scale=3,
                       n_estimators=25, data_reduction=3)
    x1 = DecisionLayer(INPUTS=x1, name="decision_3", kernel=(5, 5), kernel_shape="ellipse", down_scale=4,
                       n_estimators=25)
    x1 = DecisionLayer(INPUTS=x1, name="decision_4", kernel=(2, 2), down_scale=20,
                       param_grid=pg.random_forrest_grid_estimators())
    x1 = DecisionLayer(INPUTS=x1, name="decision_5", kernel=(5, 5), kernel_shape="ellipse", down_scale=4,
                       n_estimators=25, data_reduction=3)
    x1 = DecisionLayer(INPUTS=x1, name="decision_6", kernel=(5, 5), kernel_shape="ellipse", down_scale=3,
                       n_estimators=25, data_reduction=3)
    x1 = DecisionLayer(INPUTS=x1, name="decision_7", kernel=(5, 5), kernel_shape="ellipse", down_scale=2,
                       n_estimators=25, data_reduction=3)
    # x1 = DecisionLayer(INPUTS=x1, name="decision_4", kernel=(5, 5), down_scale=4)
    # x1 = DecisionLayer(INPUTS=x1, name="decision_5", kernel=(5, 5), down_scale=3)
    # x1 = DecisionLayer(INPUTS=x1, name="decision_6", kernel=(5, 5), down_scale=2)

    # x1 = GlobalContextLayer(INPUTS=x1, name="glob_context", down_scale=2)

    f1 = DecisionLayer(INPUTS=x1,
                       name="final_decision",
                       kernel=(3, 3),
                       kernel_shape="ellipse",
                       down_scale=2,
                       n_estimators=500)

    model = Model(graph=f1)

    d_set = DataSet(df, color_coding)
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
