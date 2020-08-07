import argparse
import os

from data_structure.segmentation_data_set import SegmentationDataSet
from structured_classifier.model import Model

from model.random_forrest import RandomStructuredRandomForrest
from model.pyramid_boosting import PyramidBoosting
from model.encoder_decoder import EncoderDecoder
from model.patch_work import PatchWork

from structured_classifier.input_layer import InputLayer
from structured_classifier.super_pixel_layer import SuperPixelLayer
from structured_classifier.graph_layer import GraphLayer
from structured_classifier.shape_refinement_layer import ShapeRefinementLayer
from structured_classifier.bottle_neck_layer import BottleNeckLayer
from structured_classifier.normalization_layer import NormalizationLayer
from structured_classifier.feature_extraction_layer import FeatureExtractionLayer

from utils import parameter_grid as pg

from utils.utils import save_dict


def main(args_):
    color_coding = {
        # "js": [[1, 1, 1], [255, 255, 0]],
        # "crack": [[255, 255, 255], [255, 0, 255]],
        # "ellipse": [[200, 0, 0], [0, 255, 255]],
        # "street_sign": [[155, 155, 155], [0, 255, 0]],
        # "man_hole": [[1, 1, 1], [0, 255, 0]],
        # "crack_cluster": [[1, 1, 1], [255, 255, 0]],
        # "crack": [[3, 3, 3], [255, 255, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        "shadow": [[1, 1, 1], [255, 0, 0]],
        # "filled_crack": [[2, 2, 2], [0, 255, 0]],
        # "lines": [[1, 1, 1], [255, 0, 0]],
        # "street": [[255, 0, 255], [255, 0, 255]],
        # "cobblestone": [[180, 50, 180], [180, 50, 180]],
        # "side_walk": [[180, 149, 200], [180, 149, 200]],
        # "vegetation": [[147, 253, 194], [147, 253, 194]],
        # "sky": [[135, 206, 255], [135, 206, 255]],
        # "human": [[199, 150, 250], [199, 150, 250]],
        # "Building": [[241, 230, 255], [241, 230, 255]],
        # "TrafficSign": [[7, 255, 255], [7, 255, 255]],
    }

    randomized_split = True
    train_test_ratio = 0.20

    df = args_.dataset_folder
    mf = args_.model_folder

    clf = "xgboost"
    opt = {
        # "layer_structure": (32, ),
        "n_estimators": 100,
        "num_parallel_tree": 5,
        }

    width = 300

    ed = EncoderDecoder(features_to_use=["hsv-color"],
                        norm_input="normalize_mean",
                        kernel_shape="ellipse",
                        clf=clf, clf_options=opt)
    x = ed.build(width=width)
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
