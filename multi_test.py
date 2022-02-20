import argparse
import os
import numpy as np
from tqdm import tqdm

from data_structure.segmentation_data_set import SegmentationDataSet
from data_structure.folder import Folder
from structured_classifier.model import Model
from utils.utils import load_dict
from data_structure.stats_handler import StatsHandler


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
    us = args_.unsupervised

    for model in os.listdir(mf):
        path = os.path.join(mf, model)
        if os.path.isfile(os.path.join(path, "report.txt")):
            continue
        if os.path.isdir(path):
            run_test(df, path, us)


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
