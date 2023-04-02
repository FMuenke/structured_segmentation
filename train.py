import argparse
import os
from time import time

from data_structure.segmentation_data_set import SegmentationDataSet
from model import RandomEnsemble, PixelSegmentor, SuperPixelSegmentor, EncoderDecoder, PyramidBoosting, Ensemble

from utils.utils import save_dict, check_n_make_dir

from test import run_test


def main(args_):
    color_coding = {
        # "0": [[0, 0, 0], [0, 0, 0]],
        "1": [[255, 255, 255], [255, 0, 0]],
        # "1": [[4, 4, 4], [255, 0, 0]],
        # "shadow": [[1, 1, 1], [0, 100, 255]]
        # "heart": [[4, 4, 4], [0, 100, 255]]
        # "nuceli": [[255, 255, 255], [100, 100, 255]],
    }

    randomized_split = True
    train_test_ratio = 0.0

    df = args_.dataset_folder

    df_train = os.path.join(df, "train")
    df_test = os.path.join(df, "test")
    mf = args_.model_folder

    downscale = 4
    sp_feature = "hsv-lm"
    ed_feature = "gray-color"
    models_to_train = {
        # "sp-1": SuperPixelSegmentor(feature_to_use=sp_feature, initial_image_down_scale=downscale),
        # "sp-2": SuperPixelSegmentor(feature_to_use=sp_feature, initial_image_down_scale=downscale),
        # "sp-3": SuperPixelSegmentor(feature_to_use=sp_feature, initial_image_down_scale=downscale),
        # "ens-1": Ensemble(features_to_use=ed_feature, initial_image_down_scale=downscale),
        # "ed-1": EncoderDecoder(features_to_use=ed_feature, initial_image_down_scale=downscale),
        "px-1": PixelSegmentor(feature_to_use=ed_feature, initial_image_down_scale=downscale, kernel=5, stride=2)
    }

    d_set = SegmentationDataSet(df_train, color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set, percentage=train_test_ratio, random=randomized_split)

    # train_set = train_set[::10]

    for model_id in models_to_train:
        model = models_to_train[model_id]
        sub_mf = os.path.join(mf, model_id)
        check_n_make_dir(sub_mf)
        t0 = time()
        model.fit(train_set)
        with open(os.path.join(sub_mf, "time.txt"), "w") as f:
            f.write("[INFO] done in %0.3fs" % (time() - t0))
        model.save(sub_mf)
        save_dict(color_coding, os.path.join(sub_mf, "color_coding.json"))
        run_test(sub_mf, df_test)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        help="Path to directory with predictions",
    )
    parser.add_argument(
        "--test_folder",
        "-tf",
        help="Path to directory with predictions",
    )
    parser.add_argument(
        "--model_folder",
        "-model",
        default="./test",
        help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
