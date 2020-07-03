import argparse
import os

from data_structure.segmentation_data_set import SegmentationDataSet
from structured_classifier.model import Model

from model.random_forrest import RandomStructuredRandomForrest
from model.pyramid_boosting import PyramidBoosting
from model.encoder_decoder import EncoderDecoder

from structured_classifier.input_layer import InputLayer
from structured_classifier.super_pixel_layer import SuperPixelLayer
from structured_classifier.decision_layer import DecisionLayer
from structured_classifier.shape_refinement_layer import ShapeRefinementLayer
from structured_classifier.normalization_layer import NormalizationLayer
from structured_classifier.feature_extraction_layer import FeatureExtractionLayer

from utils import parameter_grid as pg

from utils.utils import save_dict


def main(args_):
    color_coding = {
        # "js": [[1, 1, 1], [255, 255, 0]],
        # "cr": [[255, 255, 255], [255, 0, 255]],
        # "ellipse": [[200, 0, 0], [0, 255, 255]],
        # "street_sign": [[155, 155, 155], [0, 255, 0]],
        # "man_hole": [[1, 1, 1], [0, 255, 0]],
        # "crack_cluster": [[1, 1, 1], [255, 255, 0]],
        # "crack": [[3, 3, 3], [255, 255, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        # "shadow": [[1, 1, 1], [255, 0, 0]],
        # "filled_crack": [[2, 2, 2], [0, 255, 0]],
        # "lines": [[1, 1, 1], [255, 0, 0]],
        "bark": [[1, 1, 1], [13, 128, 228]],
        "wood": [[2, 2, 2], [144, 230, 85]],
        "textile": [[3, 3, 3], [122, 200, 59]],
        "man-made": [[4, 4, 4], [5, 56, 152]],
        "plants": [[5, 5, 5], [51, 122, 78]],
        "flowers": [[6, 6, 6], [193, 85, 180]],
        "nature": [[7, 7, 7], [192, 87, 160]],
        "glass": [[8, 8, 8], [226, 178, 123]],
        "rock": [[9, 9, 9], [17, 50, 20]],
        "stone": [[10, 10, 10], [70, 84, 67]],
    }

    randomized_split = True
    train_test_ratio = 0.20

    df = args_.dataset_folder
    mf = args_.model_folder

    clf = "xgboost"
    opt = {
        "layer_structure": (126, 32, ),
        "n_estimators": 500,
        "num_parallel_tree": 5,
        "base_estimator": {"type": "rf"},
        }
    width = 400

    x = InputLayer(name="input_interes", width=width, features_to_use=["rgb-color", "gray-lbp"])
    x = SuperPixelLayer(INPUTS=x, name="sp",
                        super_pixel_method="patches", down_scale=1,
                        feature_aggregation="hist32", clf=clf, clf_options=opt)

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
        "--model_folder", "-m", default="./test", help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
