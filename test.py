import argparse
import os

from structured_segmentation.data_structure import SegmentationDataSet
from structured_segmentation.model import Graph
from structured_segmentation.utils.utils import load_dict


def run_test(mf, df, us=False):
    color_coding = load_dict(os.path.join(mf, "color_coding.json"))
    model = Graph(mf)
    model.load(mf)
    d_set = SegmentationDataSet(df, color_coding)
    test_set = d_set.get_data()
    model.evaluate(test_set, color_coding, mf, is_unsupervised=us)


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    us = args_.unsupervised
    run_test(mf, df, us)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with images and labels folder",
    )
    parser.add_argument(
        "--model_folder", "-model", default="./test", help="Path to model directory"
    )
    parser.add_argument(
        "--unsupervised", "-unsup", type=bool, default=False, help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
