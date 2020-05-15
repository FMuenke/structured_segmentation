import argparse
import os

from data_structure.segmentation_data_set import SegmentationDataSet
from structured_classifier.model import Model

from elements.random_forrest import RandomStructuredRandomForrest
from elements.pyramid_boosting import PyramidBoosting

from elements.base_structures import u_layer, up_pyramid, down_pyramid

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

    randomized_split = False
    train_test_ratio = 0.20

    df = args_.dataset_folder
    mf = args_.model_folder

    rf = RandomStructuredRandomForrest(n_estimators=10,
                                       max_depth=2,
                                       max_kernel_sum=15,
                                       max_down_scale=6,
                                       features_to_use=["hsv-color"],
                                       clf="lr",
                                       norm_input=["normalize_mean_axis"])
    x1 = rf.build(width=600)

    pb = PyramidBoosting(n_estimators=1,
                         max_depth=5,
                         max_kernel_sum=5,
                         features_to_use="hsv-color",
                         norm_input="normalize_mean",
                         clf="lr")

    x1 = pb.build(width=300)

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
