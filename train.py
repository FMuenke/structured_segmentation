import argparse
import os
from time import time

from structured_segmentation.data_structure import SegmentationDataSet
from structured_segmentation.model import EncoderDecoder

from structured_segmentation.utils.utils import save_dict, check_n_make_dir


def main(args_):
    color_coding = {
        "class_1": [[255, 255, 255], [255, 0, 0]],
        "class_2": [[255, 0, 0], [0, 0, 255]],
        "unlabeled": [[100, 100, 100], [0, 0, 0]]
        # We can define a class explicitly as unlabeled, its pixels will not be used
    }

    df = args_.dataset_folder

    df_train = os.path.join(df, "train")
    df_test = os.path.join(df, "test")
    mf = args_.model_folder

    train_set = SegmentationDataSet(df_train, color_coding)
    train_set = train_set.get_data()

    test_set = SegmentationDataSet(df_test, color_coding)
    test_tags = test_set.get_data()

    model = EncoderDecoder(image_width=256, image_height=256)
    check_n_make_dir(mf)
    t0 = time()
    model.fit(train_set)
    with open(os.path.join(mf, "time.txt"), "w") as f:
        f.write("[INFO] Training done in %0.3fs" % (time() - t0))
    model.save(mf)
    save_dict(color_coding, os.path.join(mf, "color_coding.json"))
    model.evaluate(test_tags, color_coding, mf, is_unsupervised=False, plot_results=True)


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
