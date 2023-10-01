"""
This modul handles the segmentation dataset
as a sum of all images available for training
"""

import os
from tqdm import tqdm
import numpy as np
from structured_segmentation.data_structure.labeled_image import LabeledImage


class SegmentationDataSet:
    def __init__(self, path_to_data_set, color_coding):
        self.path_to_data_set = path_to_data_set
        self.color_coding = color_coding
        self.tag_set = self.load()

    def _load(self, tag_set, summary, path):
        print("[INFO] Try Loading Data from: {}".format(path))
        if os.path.isdir(os.path.join(path, "images")):
            print("[INFO] loading...")
            for img_f in tqdm(sorted(os.listdir(os.path.join(path, "images")))):
                if img_f.endswith((".jpg", ".png", ".tif", ".tiff", ".ppm")):
                    tag_set[len(tag_set)] = LabeledImage(
                        base_path=path,
                        data_id=img_f[:-4],
                        color_coding=self.color_coding
                    )
                    unique, counts = tag_set[len(tag_set)-1].summary()
                    for u, c in zip(unique, counts):
                        if u not in summary:
                            summary[u] = c
                        else:
                            summary[u] += c
        else:
            for f in os.listdir(str(path)):
                file_name = os.path.join(str(path), f)
                if os.path.isdir(file_name):
                    self._load(tag_set, summary, file_name)

    def load(self):
        tag_set = dict()
        summary = dict()
        self._load(tag_set, summary, self.path_to_data_set)

        print("[INFO] DataSet Summary:")
        tot = 0
        for u in summary:
            tot += summary[u]
        for u in summary:
            print("[INFO] ClassIdx {}: {}".format(u, summary[u]/tot))

        assert len(tag_set) > 0, "[ERROR] No Data was found.."
        return tag_set

    def get_data(self, percentage=0.0, random=True):
        tag_set = self.tag_set
        train_set = []
        validation_set = []
        if random:
            dist = np.random.permutation(len(tag_set))
        else:
            dist = range(len(tag_set))
        for d in dist:
            if len(validation_set) >= percentage * len(tag_set):
                train_set.append(tag_set[d])
            else:
                validation_set.append(tag_set[d])
        if percentage > 0:
            print("[INFO] Training Samples: {} / Validation Samples: {}".format(len(train_set), len(validation_set)))
            return np.array(train_set), np.array(validation_set)
        else:
            return np.array(train_set)
