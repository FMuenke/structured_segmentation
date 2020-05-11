import argparse
import os

from data_structure.segmentation_data_set import SegmentationDataSet
from structured_classifier.model import Model

from structured_classifier.decision_layer import DecisionLayer
from structured_classifier.input_layer import InputLayer
from structured_classifier.global_context_layer import GlobalContextLayer
from structured_classifier.normalization_layer import NormalizationLayer

from base_elements.base_structures import u_layer, up_pyramid, down_pyramid

from utils import parameter_grid as pg

from utils.utils import save_dict


def main(args_):
    color_coding = {
        # "man_hole": [[1, 1, 1], [0, 255, 0]],
        # "crack_cluster": [[1, 1, 1], [255, 255, 0]],
        # "crack": [[3, 3, 3], [255, 255, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        "shadow": [[1, 1, 1], [255, 0, 0]],
        # "filled_crack": [[2, 2, 2], [0, 255, 0]],
    }

    randomized_split = True
    train_test_ratio = 0.20

    df = args_.dataset_folder
    mf = args_.model_folder

    clf = "b_rf"
    clf_opt = {"n_estimators": 50}

    x1 = InputLayer("input_1", ["hsv-color"], width=300)
    x1 = NormalizationLayer(INPUTS=x1, name="norm_1", norm_option="normalize_mean")

    x1 = u_layer(x1, "u_s_clf_1", depth=4, repeat=1, kernel=(5, 5), clf=clf, clf_options=clf_opt)

    # x12 = u_layer(x1, "u_s_clf_2", depth=3, repeat=1, kernel=(3, 3), clf="b_rf", clf_options=clf_opt)

    f1 = DecisionLayer(INPUTS=x1,
                       name="final_decision",
                       kernel=(5, 5),
                       kernel_shape="ellipse",
                       clf=clf,
                       clf_options=clf_opt,
                       down_scale=1,
                       )

    model = Model(graph=x1)

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
