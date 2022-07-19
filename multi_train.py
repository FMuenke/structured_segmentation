import argparse
import os
import numpy as np
from tqdm import tqdm

from data_structure.segmentation_data_set import SegmentationDataSet
from structured_classifier.model import Model
from data_structure.stats_handler import StatsHandler
from data_structure.folder import Folder

from structured_classifier import InputLayer
from structured_classifier import CIPPLayer
from structured_classifier import PixelLayer

from utils.utils import save_dict, load_dict


def model_v1():
    x = InputLayer("IN", features_to_use="RGB-color", initial_down_scale=1)
    x = CIPPLayer(x, "SIMPLE", operations=[
        "blurring",
        "invert",
        "edge",
        "closing",
        "erode",
    ], selected_layer=[0, 1, 2], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)
    return model


def model_cipp():
    x = InputLayer("IN", features_to_use="RGB-color", initial_down_scale=1)
    x = CIPPLayer(x, "CIPP", operations=[
        "blurring",
        "invert",
        ["threshold", "threshold_otsu", "edge"],
        "closing",
        "erode",
    ], selected_layer=[1], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)
    return model


def model_v2():
    x = InputLayer("IN", features_to_use="gray-color", initial_down_scale=1)
    x = CIPPLayer(x, "SIMPLE", operations=[
        "blurring",
        "top_clipping_percentile",
        "negative_closing",
        "threshold",
        "remove_small_objects",
    ], selected_layer=[0], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)
    return model


def model_cipp_watershed():
    x = InputLayer("IN", features_to_use="RGB-color", initial_down_scale=1)
    x = CIPPLayer(x, "SIMPLE", operations=[
        # "blurring",
        "invert",
        "watershed",
        "closing",
        "erode",
    ], selected_layer=[0, 1, 2], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)
    return model


def model_v4():
    x = InputLayer("IN", features_to_use=["RGB-color"], initial_down_scale=1)
    x = CIPPLayer(x, "SIMPLE", operations=[
        "watershed",
        "remove_small_holes",
        "remove_small_objects",
    ], selected_layer=[1], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)
    return model


def model_px():
    x = InputLayer("IN", features_to_use=["RGB-lm"], initial_down_scale=1)
    x = PixelLayer(x, "PX", clf="rf")
    model = Model(graph=x)
    return model


def convert_cls_to_color(cls_map, color_coding, unsupervised=False):
    h, w = cls_map.shape[:2]
    color_map = np.zeros((h, w, 3))
    if unsupervised:
        unique_y = np.unique(cls_map)
        for u in unique_y:
            if str(u) not in color_coding:
                color_coding[str(u)] = [[0, 0, 0],
                                        [np.random.randint(255), np.random.randint(255), np.random.randint(255)]]
    for idx, cls in enumerate(color_coding):
        iy, ix = np.where(cls_map == idx + 1)
        color_map[iy, ix, :] = [color_coding[cls][1][2],
                                color_coding[cls][1][1],
                                color_coding[cls][1][0]]
    return color_map


def run_training(df, mf, number_of_tags):
    print(mf)
    color_coding = {
        "ellipse": [[255, 255, 255], [255, 0, 0]],
    }

    randomized_split = True
    train_test_ratio = 0.5

    # DEFINE MODEL ###############
    model = model_cipp()
    ##############################

    d_set = SegmentationDataSet(df, color_coding)
    tag_set = d_set.load()
    # train_set, validation_set = d_set.split(tag_set, percentage=train_test_ratio, random=randomized_split)

    if number_of_tags != 0:
        tags = [tag_set[t] for t in tag_set]
        half = len(tags) // 2
        train_tags = tags[:half]
        test_tags = tags[half:]
        print("Number of training and validation samples - {}/{}".format(len(tags), len(tags)))
        seed_id = int(mf.split("-RUN-")[-1])
        rng = np.random.default_rng(seed_id)
        train_set = rng.choice(train_tags, number_of_tags, replace=False)
        # number_of_tags = np.min([number_of_tags, len(tags)-1])
        validation_set = rng.choice(test_tags, number_of_tags, replace=False)
        print("Number of training images reduced! - {}/{} -".format(len(train_set), len(validation_set)))
        model.fit(train_set, validation_set)
        model.save(mf)
    else:
        train_set, validation_set = d_set.split(tag_set, percentage=train_test_ratio, random=randomized_split)
        model.fit(train_set, validation_set)
        model.save(mf)

    save_dict(color_coding, os.path.join(mf, "color_coding.json"))


def run_test(df, mf, us):
    color_coding = load_dict(os.path.join(mf, "color_coding.json"))

    model = Model(mf)
    model.load(mf)

    res_fol = Folder(os.path.join(mf, "segmentations"))
    res_fol.check_n_make_dir(clean=True)

    vis_fol = Folder(os.path.join(mf, "overlays"))
    vis_fol.check_n_make_dir(clean=True)

    d_set = SegmentationDataSet(df, color_coding)
    t_set = d_set.load()

    sh = StatsHandler(color_coding)
    print("Processing Images...")
    for tid in tqdm(t_set):
        cls_map = model.predict(t_set[tid].load_x())
        color_map = convert_cls_to_color(cls_map, color_coding, unsupervised=us)
        t_set[tid].write_result(res_fol.path(), color_map)
        if not us:
            t_set[tid].eval(color_map, sh)
        t_set[tid].visualize_result(vis_fol.path(), color_map)

    sh.eval()
    sh.show()
    sh.write_report(os.path.join(mf, "report.txt"))


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder

    if not os.path.isdir(mf):
        os.mkdir(mf)

    number_of_images = [1, 2, 4, 8, 16, 32, 64, 128]
    iterations = 20

    for n in number_of_images:
        for i in range(iterations):
            sub_mf = os.path.join(mf, "-{}-RUN-{}".format(n, i))
            if not os.path.isdir(os.path.join(mf, "-{}-RUN-{}".format(n, i))):
                run_training(os.path.join(df, "train"), sub_mf, n)
            if not os.path.isfile(os.path.join(sub_mf, "report.txt")):
                run_test(os.path.join(df, "test"), sub_mf, False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
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
