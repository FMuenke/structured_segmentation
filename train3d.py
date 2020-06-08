import argparse
import os

from data_structure.video_set import VideoSet
from structured_classifier.model import Model

from structured_classifier.decision_3d_layer import Decision3DLayer
from structured_classifier.input_3d_layer import Input3DLayer
from structured_classifier.voting_3d_layer import Voting3DLayer

from elements.random_forrest import RandomStructuredRandomForrest3D
from elements.pyramid_boosting import PyramidBoosting3D

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

    # rf = RandomStructuredRandomForrest3D(n_estimators=100, max_down_scale=3, max_depth=2)
    # x = rf.build(initial_down_scale=1)

    pb = PyramidBoosting3D(n_estimators=1, max_depth=2, max_kernel_sum=5)
    x = pb.build(initial_down_scale=2)

    model = Model(graph=x)

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
