import argparse
import os

from data_structure.video_set import VideoSet
from structured_classifier.model import Model

from structured_classifier.decision_3d_layer import Decision3DLayer
from structured_classifier.input_3d_layer import Input3DLayer
from structured_classifier.voting_3d_layer import Voting3DLayer
from structured_classifier.super_pixel_3d_layer import SuperPixel3DLayer
from structured_classifier.shape_refinement_3d_layer import ShapeRefinement3DLayer

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

    rf = RandomStructuredRandomForrest3D(n_estimators=25, features_to_use=["gray-lbp"],
                                         kernel_shape="ellipse",
                                         max_down_scale=5, max_depth=1, max_kernel_sum=25,
                                         clf="b_rf", clf_options={"n_estimators": 100})
    x = rf.build(width=300, output_option="boosting")

    # pb = PyramidBoosting3D(n_estimators=3, max_depth=1, max_kernel_sum=5, data_reduction=6, features_to_use="gray-lbp")
    # x = pb.build(width=300)

    # x_in = Input3DLayer(name="in", features_to_use="gray-lbp", width=300)
    x = ShapeRefinement3DLayer(INPUTS=x, global_kernel=(3, 9, 9), shape="ellipse", name="sr")
    # x1 = SuperPixel3DLayer(INPUTS=x_in, name="sp1", time_range=5, super_pixel_method="slic", option=200)
    # x2 = SuperPixel3DLayer(INPUTS=x_in, name="sp2", time_range=11, super_pixel_method="slic", option=50)
    # x3 = SuperPixel3DLayer(INPUTS=x_in, name="sp3", time_range=7, super_pixel_method="slic", option=100)
    # x = SuperPixel3DLayer(INPUTS=x, name="sp2", time_range=11, super_pixel_method="slic", option=100)
    # x = SuperPixel3DLayer(INPUTS=x, name="sp3", time_range= 5, super_pixel_method="slic", option= 50)
    # x = Decision3DLayer(INPUTS=[x1, x2, x3], kernel=(15, 3, 3), name="dl-1", down_scale=0, data_reduction=3)
    # x = Decision3DLayer(INPUTS=x, kernel=(1, 5, 5), name="dl-2", down_scale=2, data_reduction=3)
    # x = Decision3DLayer(INPUTS=x, kernel=(1, 5, 5), name="dl-3", down_scale=3, data_reduction=3)
    # x = Decision3DLayer(INPUTS=x, kernel=(1, 5, 5), name="dl-4", down_scale=2, data_reduction=3)
    # x = Decision3DLayer(INPUTS=x, kernel=(1, 3, 3), name="dl-5", down_scale=0)

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
