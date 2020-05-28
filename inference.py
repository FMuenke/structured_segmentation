import argparse
import os
import numpy as np
from tqdm import tqdm

import cv2

from data_structure.segmentation_data_set import SegmentationDataSet
from data_structure.folder import Folder
from structured_classifier.model import Model
from utils.utils import load_dict
from data_structure.stats_handler import StatsHandler


def convert_cls_to_color(cls_map, color_coding):
    h, w = cls_map.shape[:2]
    color_map = np.zeros((h, w, 3))
    clmp = dict()
    for idx, cls in enumerate(color_coding):
        clmp[idx + 1] = cls
    for x in range(w):
        for y in range(h):
            if int(cls_map[y, x]) in clmp:
                color_map[y, x, :] = color_coding[clmp[int(cls_map[y, x])]][1]
    return color_map


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder

    color_coding = load_dict(os.path.join(mf, "color_coding.json"))

    model = Model(mf)
    model.load(mf)

    res_fol = Folder(os.path.join(df, "inference"))
    res_fol.check_n_make_dir(clean=True)

    d_set = SegmentationDataSet(df, color_coding)
    t_set = d_set.load()

    sh = StatsHandler(color_coding)
    print("Processing Images...")
    for tid in tqdm(t_set):
        cls_map = model.predict(t_set[tid].load_x())
        color_map = convert_cls_to_color(cls_map, color_coding)
        im_id = os.path.basename(t_set[tid].path_to_image_file)
        cv2.imwrite(os.path.join(str(res_fol), im_id[:-4] + ".png"), color_map)

    sh.eval()
    sh.show()
    sh.write_report(os.path.join(mf, "report.txt"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with images and labels folder",
    )
    parser.add_argument(
        "--model_folder", "-m", default="./test", help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
