import argparse
import os

from data_structure.video_set import VideoSet
from structured_classifier.model import Model

from structured_classifier.decision_3d_layer import Decision3DLayer
from structured_classifier.input_3d_layer import Input3DLayer
from structured_classifier.voting_3d_layer import Voting3DLayer
from structured_classifier.super_pixel_3d_layer import SuperPixel3DLayer
from structured_classifier.bottle_neck_3d_layer import BottleNeck3DLayer
from structured_classifier.shape_refinement_3d_layer import ShapeRefinement3DLayer

from model.random_forrest import RandomStructuredRandomForrest3D
from model.pyramid_boosting import PyramidBoosting3D
from model.encoder_decoder import EncoderDecoder3D

from utils import parameter_grid as pg

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

    clf = "b_rf"
    opt = {"n_estimators": 500,
           "num_parallel_tree": 5,
           "layer_structure": (126, 32, )}

    rf = RandomStructuredRandomForrest3D(n_estimators=25, features_to_use=["gray-lbp"],
                                         kernel_shape="square",
                                         max_down_scale=3, max_depth=1, max_kernel_sum=25,
                                         clf=clf, clf_options=opt)
    x = rf.build(width=64, height=64, output_option="boosting")

    x_in = Input3DLayer(name="in", features_to_use="gray-lbp", width=300)
    x1 = SuperPixel3DLayer(INPUTS=x_in, name="pa0", time_range=5,
                           super_pixel_method="patches", down_scale=2,
                           feature_aggregation="hist25", clf=clf, clf_options=opt)
    x2 = SuperPixel3DLayer(INPUTS=x_in, name="pa1", time_range=5,
                           super_pixel_method="quickshift", down_scale=1,
                           feature_aggregation="hist25", clf=clf, clf_options=opt)
    x3 = SuperPixel3DLayer(INPUTS=x_in, name="pa2", time_range=5,
                           super_pixel_method="quickshift", down_scale=2,
                           feature_aggregation="hist25", clf=clf, clf_options=opt)

    x = BottleNeck3DLayer(INPUTS=[x1, x2, x3], name="bott")
    x_in2 = Input3DLayer(name="in2", features_to_use="gray-lbp", width=300)
    x = Decision3DLayer(INPUTS=[x, x_in2], name="merge", clf=clf, clf_options=opt, kernel=(1, 5, 5))

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
