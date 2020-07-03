import argparse
import os
import numpy as np
from PIL import Image


def read_classification_label_file(path_to_label_file):
    cls_list = []
    with open(path_to_label_file) as f:
        for line in f:
            info = line.split(" ")
            cls_list.append(info[0].replace("\n", ""))
    if len(cls_list) == 0:
        return ["bg"]
    else:
        return cls_list


def label_folder(clmp, folder):
    im_dir = os.path.join(folder, "images")
    lb_dir = os.path.join(folder, "labels")
    for lb in os.listdir(lb_dir):
        lb_f = os.path.join(lb_dir, lb)
        if os.path.isfile(lb_f) and lb.endswith(".txt"):
            labels = read_classification_label_file(lb_f)
            label = labels[0]

            if label not in clmp:
                clmp[label] = [[len(clmp) + 1, len(clmp) + 1, len(clmp) + 1], [
                    np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
                ]]
            im = Image.open(os.path.join(im_dir, lb[:-4] + ".png"))
            w, h = im.size
            label_map = np.ones((h, w, 3), dtype=np.uint8) * clmp[label][0][0]
            label_map = Image.fromarray(label_map)
            label_map.save(os.path.join(lb_dir, lb[:-4] + ".png"))


def main(args_):
    clmp = {}
    df = args_.dataset_folder
    for d in os.listdir(df):
        c_df = os.path.join(df, d)
        if os.path.isdir(c_df):
            label_folder(clmp, c_df)

    print(clmp)
    s = ""
    for i, label in enumerate(clmp):

        s += "\"{}\": [{}, {}],\n".format(label, clmp[label][0], clmp[label][1])

    with open(os.path.join(df, "seg_info.txt"), "w") as f:
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
