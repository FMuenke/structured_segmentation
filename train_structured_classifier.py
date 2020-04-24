import argparse

from data_structure.data_set import DataSet
from structured_classifier.structured_classifier import StructuredClassifier

from options.config import build_cfg, save_config


def train_instance(mf, cfg, train_set, validation_set):
    save_config(mf, cfg)

    s_clf = StructuredClassifier(mf)
    s_clf.build(cfg)

    s_clf.fit(train_set, validation_set, data_reduction_factor=cfg.data_reduction_factor)
    s_clf.save()


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder

    cfg = build_cfg()
    d_set = DataSet(df, cfg.color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set, percentage=cfg.train_test_ratio, random=cfg.randomized_split)
    train_instance(mf, cfg, train_set, validation_set)


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
