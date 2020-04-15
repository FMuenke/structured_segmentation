import argparse
import os
import numpy as np
from tqdm import tqdm

from data_structure.data_set import DataSet
from data_structure.folder import Folder
from structured_classifier.structured_classifier import StructuredClassifier
from options.config import load_config
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

    cfg = load_config(mf)

    s_clf = StructuredClassifier(mf)
    s_clf.load()

    res_fol = Folder(os.path.join(mf, "segmentations"))
    res_fol.check_n_make_dir(clean=True)

    vis_fol = Folder(os.path.join(mf, "overlays"))
    vis_fol.check_n_make_dir(clean=True)

    d_set = DataSet(df, cfg.color_coding)
    t_set = d_set.load()

    sh = StatsHandler(cfg.color_coding)
    for tid in tqdm(t_set):
        cls_map = s_clf.inference(t_set[tid].load_x())
        color_map = convert_cls_to_color(cls_map, cfg.color_coding)
        t_set[tid].write_result(res_fol.path(), color_map)
        t_set[tid].eval(color_map, sh)
        t_set[tid].visualize_result(vis_fol.path(), color_map)

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
