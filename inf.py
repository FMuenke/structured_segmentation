import argparse
import os
import numpy as np
from tqdm import tqdm
import cv2

from data_structure.folder import Folder
from structured_classifier.model import Model


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


def run_inference_on_folder(df, model, model_id):

    if not os.path.isdir(os.path.join(df, "images")):
        return None

    res_fol = Folder(os.path.join(df, "inference_by_{}".format(model_id)))
    res_fol.check_n_make_dir(clean=True)

    print("Processing Images...")
    for tid in tqdm(os.listdir(os.path.join(df, "images"))):
        if tid.endswith(".jpg"):
            if not os.path.isfile(os.path.join(str(res_fol), tid[:-4] + ".png")):
                im = cv2.imread(os.path.join(os.path.join(df, "images"), tid))
                cls_map = model.predict(im)
                cv2.imwrite(os.path.join(str(res_fol), tid[:-4] + ".png"), cls_map)


def main(args_):
    df = args_.data_folder
    mf = args_.model_folder

    model = Model(mf)
    model.load(mf)

    if mf.endswith("/"):
        mf = mf[:-1]
    model_id = os.path.basename(mf)

    run_inference_on_folder(df, model, model_id)

    for sub_df in os.listdir(df):
        run_inference_on_folder(os.path.join(df, sub_df), model, model_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
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
