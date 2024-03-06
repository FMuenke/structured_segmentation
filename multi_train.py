import argparse
import os
from time import time
import numpy as np
from tqdm import tqdm
import shutil
import logging

from structured_segmentation.data_structure.segmentation_data_set import SegmentationDataSet
from structured_segmentation.model import Graph
from structured_segmentation.model import EncoderDecoder, SuperPixelSegmentor
from structured_segmentation.model.pixel_segmentor import PixelSegmentor

from structured_segmentation.utils.utils import save_dict, load_dict, check_n_make_dir


def run_training(df, mf, number_of_tags):
    print(mf)
    color_coding = {
        # "crack": [[1, 1, 1], [0, 255, 0]],
        # "crack": [[255, 255, 255], [0, 255, 0]],
        # "crack": [[0, 255, 0], [0, 255, 0]], 
        # "pothole": [[255, 0, 0], [255, 0, 0]],
        # "1": [[255, 85, 0], [255, 85, 0]],
        # "2": [[255, 170, 127], [255, 170, 127]],
        # "3": [[255, 255, 127], [255, 255, 127]],
        # "4": [[85, 85, 255], [85, 85, 255]],
        # "5": [[255, 255, 255], [255, 255, 255]],
        # "6": [[170, 0, 127], [170, 0, 127]],
        # "7": [[85, 170, 127], [85, 170, 127]],
        # "8": [[255, 85, 255], [255, 85, 255]],
        # "9": [[255, 0, 0], [255, 0, 0]],
        # "10": [[0, 0, 127], [0, 0, 127]],
        # "11": [[170, 0, 0], [170, 0, 0]],
        # "0": [[128, 0, 0], [128, 0, 0]],
        # "1": [[0, 128, 0], [0, 128, 0]],
        # "2": [[0, 255, 0], [0, 255, 0]],
        # "3": [[128, 128, 128], [128, 128, 128]],
        # "unlabeled": [[255, 255, 255], [255, 255, 255]],
        # "5": [[0, 0, 255], [0, 0, 255]],
        # "6": [[255, 0, 0], [255, 0, 0]],
        'Building-flooded':[[1, 1, 1], [255, 0, 0]], 
        'Building-non-flooded': [[2, 2, 2], [0, 255, 0]], 
        'Road-flooded': [[3, 3, 3], [0, 0, 255]],
        'Road-non-flooded': [[4, 4, 4], [255, 255, 0]], 
        'Water': [[5, 5, 5], [255, 0, 255]], 
        'Tree': [[6, 6, 6], [0, 255, 255]],
        'Vehicle': [[7, 7, 7], [100, 100, 100]], 
        'Pool':[[8, 8, 8], [255, 255, 255]], 
        'Grass': [[9, 9, 9], [0, 100, 0]],
    }
    ""

    lib_data_redu = {2: 0, 4: 0, 8: 0.1, 16: 0.2, 32: 0.4, 64:  0.6, 128: 0.6, 256: 0.66, 512: 0.80, 1024: 0.80}
    data_reduction = lib_data_redu[number_of_tags]

    # DEFINE MODEL ###############
    model = EncoderDecoder(data_reduction=data_reduction, image_height=256, image_width=256)
    # model = SuperPixelSegmentor(image_height=256, image_width=256)
    ##############################

    d_set = SegmentationDataSet(df, color_coding)
    tags = d_set.get_data()

    t0 = time()
    if number_of_tags != 0:
        tags = d_set.get_data()
        print("Number of training samples - {}".format(len(tags)))
        seed_id = int(mf.split("-RUN-")[-1])
        rng = np.random.default_rng(seed_id)
        train_set = list(rng.choice(tags, number_of_tags, replace=False))
        print("Number of training images reduced! - {} -".format(len(train_set)))
        model.fit(train_set, None)
        model.save(mf)
    else:
        train_set, validation_set = d_set.get_data(percentage=0.2, random=True)
        model.fit(train_set, validation_set)
        model.save(mf)
    save_dict({"train_time": time() - t0}, os.path.join(mf, "train_time_log.json"))
    save_dict(color_coding, os.path.join(mf, "color_coding.json"))


def run_test(df, mf, us):
    color_coding = load_dict(os.path.join(mf, "color_coding.json"))

    model = Graph(mf)
    model.load(mf)

    res_fol = os.path.join(mf, "segmentations")
    check_n_make_dir(res_fol, clean=True)

    vis_fol = os.path.join(mf, "overlays")
    check_n_make_dir(vis_fol, clean=True)

    d_set = SegmentationDataSet(df, color_coding)
    t_set = d_set.get_data()

    model.evaluate(t_set, color_coding, mf, plot_results=False)
    # shutil.make_archive(os.path.join(mf, "graph"), 'zip', os.path.join(mf, "graph"))
    shutil.rmtree(os.path.join(mf, "graph"))


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder

    if not os.path.isdir(mf):
        os.mkdir(mf)

    number_of_images = [4, 8, 16, 32, 64, 128, 256]  # 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    iterations = 10

    for n in number_of_images:
        for i in range(iterations):
            sub_mf = os.path.join(mf, "-{}-RUN-{}".format(n, i))
            if not os.path.isdir(os.path.join(mf, "-{}-RUN-{}".format(n, i))):
                run_training(os.path.join(df, "train"), sub_mf, n)
            if not os.path.isfile(os.path.join(sub_mf, "report.txt")):
                run_test(os.path.join(df, "test"), sub_mf, False)


def parse_args():
    logging.basicConfig(level=logging.ERROR)
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
