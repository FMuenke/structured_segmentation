import argparse

from data_structure.data_set import DataSet
from structured_classifier.structured_classifier import StructuredClassifier
from structured_classifier.structured_boosting import StructuredBoosting

from options.boosting_config import build_cfg, save_config


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    bm = args_.base_model

    base_model = StructuredClassifier(bm)
    base_model.load()

    cfg = build_cfg()
    d_set = DataSet(df, cfg.color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set, percentage=cfg.train_test_ratio, random=cfg.randomized_split)

    save_config(mf, cfg)

    s_clf = StructuredBoosting(mf)
    s_clf.build(cfg, base_model)

    s_clf.fit(train_set, validation_set, data_reduction_factor=cfg.data_reduction_factor)
    s_clf.save()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with training data",
    )
    parser.add_argument(
        "--base_model",
        "-bm",
        default="./test",
        help="Path to directory with base model",
    )
    parser.add_argument(
        "--model_folder", "-m", default="./test", help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
