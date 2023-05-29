import argparse
import os
from time import time

from data_structure.segmentation_data_set import SegmentationDataSet
from model import EncoderDecoder, PyramidBoosting

from utils.utils import save_dict, check_n_make_dir


def main(args_):
    color_coding = {
        # "0": [[0, 0, 0], [0, 0, 0]],
        "1": [[255, 255, 255], [255, 0, 0]],
        # "1": [[4, 4, 4], [255, 0, 0]],
        # "shadow": [[1, 1, 1], [0, 100, 255]]
        # "heart": [[4, 4, 4], [0, 100, 255]]
        # "nuceli": [[255, 255, 255], [100, 100, 255]],
    }

    df = args_.dataset_folder

    df_train = os.path.join(df, "train")
    df_test = os.path.join(df, "test")
    mf = args_.model_folder

    im_h, im_w, downscale = 256, 256, None
    models_to_train = {
        # "ed-hsv-1": EncoderDecoder(im_h, im_w, downscale, features_to_use="hsv-color"),
        # "ed-rgb-1": EncoderDecoder(im_h, im_w, downscale, features_to_use="rgb-color")
        # "ed-opp-1": EncoderDecoder(im_h, im_w, downscale, features_to_use="opponent-color"),
        "ed-RGB-1": EncoderDecoder(im_h, im_w, downscale, features_to_use="RGB-color"),
        "ed-RGB-2": EncoderDecoder(im_h, im_w, downscale, features_to_use="RGB-color"),
        "ed-RGB-3": EncoderDecoder(im_h, im_w, downscale, features_to_use="RGB-color"),
    }

    train_set = SegmentationDataSet(df_train, color_coding)
    train_set = train_set.get_data()

    test_set = SegmentationDataSet(df_test, color_coding)
    test_tags = test_set.get_data()

    for model_id in models_to_train:
        model = models_to_train[model_id]
        sub_mf = os.path.join(mf, model_id)
        check_n_make_dir(sub_mf)
        t0 = time()
        model.fit(train_set)
        with open(os.path.join(sub_mf, "time.txt"), "w") as f:
            f.write("[INFO] Training done in %0.3fs" % (time() - t0))
        model.save(sub_mf)
        save_dict(color_coding, os.path.join(sub_mf, "color_coding.json"))
        model.evaluate(test_tags, color_coding, sub_mf, is_unsupervised=False, plot_results=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
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
