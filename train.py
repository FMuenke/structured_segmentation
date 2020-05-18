import argparse
import os

from data_structure.segmentation_data_set import SegmentationDataSet
from structured_classifier.model import Model

from elements.random_forrest import RandomStructuredRandomForrest
from elements.pyramid_boosting import PyramidBoosting
from elements.encoder_decoder import EncoderDecoder

from structured_classifier.input_layer import InputLayer
from structured_classifier.decision_layer import DecisionLayer
from structured_classifier.shape_refinement_layer import ShapeRefinementLayer

from elements.base_structures import u_layer, up_pyramid, down_pyramid

from utils import parameter_grid as pg

from utils.utils import save_dict


def main(args_):
    color_coding = {
        "ellipse": [[200, 0, 0], [0, 255, 255]]
        # "street_sign": [[155, 155, 155], [0, 255, 0]]
        # "man_hole": [[1, 1, 1], [0, 255, 0]],
        # "crack_cluster": [[1, 1, 1], [255, 255, 0]],
        # "crack": [[3, 3, 3], [255, 255, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        # "shadow": [[1, 1, 1], [255, 0, 0]],
        # "filled_crack": [[2, 2, 2], [0, 255, 0]],
    }

    randomized_split = True
    train_test_ratio = 0.20

    df = args_.dataset_folder
    mf = args_.model_folder

    x = InputLayer(name="in", features_to_use="gray-color")
    # x = DecisionLayer(INPUTS=x, name="dec_1", kernel=(20, 20), kernel_shape="cross", down_scale=2, clf="lr")
    x = ShapeRefinementLayer(INPUTS=x, name="shape", shape="rectangle")

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
