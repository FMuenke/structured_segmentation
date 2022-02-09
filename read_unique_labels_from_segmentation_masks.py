import argparse
import os
import numpy as np
import cv2
from multiprocessing.pool import Pool


def read_unique(path_to_label_file):
    lbm = cv2.imread(path_to_label_file)
    if lbm is None:
        print("ALERT {} is NONE".format(path_to_label_file))
        return []
    height, width = lbm.shape[:2]
    lbm = np.reshape(lbm, (height * width, 3))
    unique_k = []
    for b, g, r in np.unique(lbm, axis=0):
        unique_k.append("[{}, {}, {}]".format(r, g, b))
    return unique_k


def read_folder(path, collector):
    if not os.path.isdir(os.path.join(path, "labels")):
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                read_folder(os.path.join(path, folder), collector)
    else:
        for label_map_file in os.listdir(os.path.join(path, "labels")):
            if not label_map_file.endswith((".png", ".tif", "tiff")):
                continue
            if not os.path.isfile(os.path.join(path, "labels", label_map_file)):
                continue
            collector.append(os.path.join(path, "labels", label_map_file))
    print("Folder: {} done.".format(path))


def main(args_):
    collector = []
    df = args_.dataset_folder
    read_folder(df, collector)
    print("Found {} files".format(len(collector)))

    with Pool() as p:
        results = p.map(read_unique, collector)

    collector_unique = {}
    for r in results:
        for k in r:
            if k not in collector_unique:
                collector_unique[k] = len(collector_unique)

    s = ""
    for i, k in enumerate(collector_unique):
        s += "\"{}\": [{}, {}],\n".format(i, k, k)

    with open(os.path.join(df, "cls_mapping.txt"), "w") as f:
        f.write(s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with predictions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
