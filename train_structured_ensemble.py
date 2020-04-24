import argparse

from data_structure.data_set import DataSet
from structured_classifier.structured_ensemble import StructuredEnsemble

from options.ensemble_config import EnsembleConfig

from utils.utils import check_n_make_dir


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    check_n_make_dir(mf)

    ensemble_cfg = EnsembleConfig()
    ensemble_cfg.build()

    d_set = DataSet(df, ensemble_cfg.color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(
        tag_set, percentage=ensemble_cfg.train_test_ratio, random=ensemble_cfg.randomized_split
    )

    s_ens = StructuredEnsemble(mf)
    s_ens.fit(train_set, validation_set, ensemble_cfg)


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
