import argparse
import os

from data_structure.video_set import VideoSet
from structured_classifier.model import Model

from structured_classifier.decision_3d_layer import Decision3DLayer
from structured_classifier.input_3d_layer import Input3DLayer
from structured_classifier.voting_3d_layer import Voting3DLayer

from utils import parameter_grid as pg

from elements.base_structures import u_layer_3d

from utils.utils import save_dict


def main(args_):
    color_coding = {
        # "man_hole": [[1, 1, 1], [0, 255, 0]],
        # "crack_cluster": [[1, 1, 1], [255, 255, 0]],
        # "crack": [[3, 3, 3], [255, 255, 0]],
        "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        # "shadow": [[1, 1, 1], [255, 0, 0]],
        # "filled_crack": [[2, 2, 2], [0, 255, 0]],
    }

    randomized_split = False
    train_test_ratio = 0.20

    df = args_.dataset_folder
    mf = args_.model_folder

    clf = "rf"
    clf_opt = {
        "n_estimators": 10,
        "layer_structure": (800, 400, 200, 100, 50, 25,),
        "max_iter": 10000000
    }

    x1 = Input3DLayer("input_1", ["gray-color"], width=150)
    x2 = Input3DLayer("input_2", ["gray-color"], width=150)

    x1 = Decision3DLayer(INPUTS=x1, name="d1", kernel=(11, 1, 1), kernel_shape="ellipse")
    x1 = Decision3DLayer(INPUTS=x1, name="d2", kernel=(5, 3, 3), kernel_shape="ellipse", down_scale=1)
    x1 = Decision3DLayer(INPUTS=x1, name="d3", kernel=(1, 3, 3), kernel_shape="ellipse", down_scale=2)

    x2 = u_layer_3d(x2, "u_structure", kernel=(1, 3, 3), depth=3)

    x3 = Decision3DLayer(INPUTS=[x1, x2], name="m1", kernel=(1, 1, 1), kernel_shape="ellipse")


    model = Model(graph=x3)

    d_set = VideoSet(df, color_coding)
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
