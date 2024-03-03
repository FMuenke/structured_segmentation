"""
This Modul contains the LabeledImage Class that
implements a trainable instance of an image and handles all image
and segmentation mask loading
"""

import os
import numpy as np
from PIL import Image
import cv2
from structured_segmentation.data_structure.image_container import ImageContainer


def get_file_name(base_path, data_id, extensions):
    """
    This function returns an existing image file provided a list of extensions
    base_path: The folder that contains the potential file (str)
    data_id: name of the file (str)
    extensions: list of str that contains all possible extensions
    """
    for ext in extensions:
        filename = os.path.join(base_path, data_id + ext)
        if os.path.isfile(filename):
            return filename
    return None


class LabeledImage:
    """
    The labeled image class is used to handle all training images
    """

    image_extensions = [".jpg", ".JPG", ".png", "PNG", ".jpeg", ".ppm", ".tif"]
    label_extensions = [".png", ".tif", "_label.tif",
                        "_label.tiff", ".tiff", ".ppm", "_label.png", "_segmentation.png",
                        "_label_ground-truth.png", "GT.png"
                        ]

    def __init__(self, base_path, data_id, color_coding, augmentations=None):
        """
        Initialize the Labeled image with the base path and the expected name
        Additionally a dictionary providing all info on expected classes and color encodings
        """
        self.id = data_id
        self.path = base_path
        self.color_coding = color_coding
        self.augmentations = augmentations

        self.image_path = os.path.join(base_path, "images")
        self.label_path = os.path.join(base_path, "labels")

        self.image_file = get_file_name(self.image_path, self.id, self.image_extensions)
        self.label_file = get_file_name(self.label_path, self.id, self.label_extensions)

    def create_augmented_tag(self, augmentations):
        """
        This function create a copy oof this tag, that contains an augmented version
        augmentations: Augmentations (Obj)
        """
        return LabeledImage(
            base_path=self.path,
            data_id=self.id,
            color_coding=self.color_coding,
            augmentations=augmentations,
        )

    def summary(self):
        """
        This function loads a summary of all contained classes in the image
        """
        y_gtr = self.load_y([100, 100])
        unique, counts = np.unique(y_gtr, return_counts=True)
        return unique, counts

    def get_image_size(self):
        """
        This function returns the image size of the attached image file
        """
        if self.image_file is None:
            raise Exception("NO IMAGE FILE AVAILABLE")
        im = Image.open(self.image_file)
        width, height = im.size
        return height, width

    def load_x(self):
        """
        This function returns the image data
        If augmentations are set, the augmentations are applied
        """
        img = cv2.imread(self.image_file)
        assert img is not None, print(f"[ERROR]: Could not load {self.image_file}")
        if self.augmentations is not None:
            img = self.augmentations.apply(img)
        return img

    def load_y_as_color_map(self, label_size):
        """
        This function reads the segmentation mask and returns the
        ground truth label_map/segmentation mask of the image
        and converts it based on the color-map
        """
        y_img = np.zeros((label_size[0], label_size[1], 3))
        if self.label_file is not None:
            lbm = cv2.imread(self.label_file)
            lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
            for _, cls in enumerate(self.color_coding):
                col_0 = np.zeros((label_size[0], label_size[1]))
                col_1 = np.zeros((label_size[0], label_size[1]))
                col_2 = np.zeros((label_size[0], label_size[1]))

                col_0[lbm[:, :, 0] == self.color_coding[cls][0][2]] = 1
                col_1[lbm[:, :, 1] == self.color_coding[cls][0][1]] = 1
                col_2[lbm[:, :, 2] == self.color_coding[cls][0][0]] = 1
                col = col_0 + col_1 + col_2
                index_y, index_x = np.where(col == 3)
                y_img[index_y, index_x, :] = [
                    self.color_coding[cls][1][2],
                    self.color_coding[cls][1][1],
                    self.color_coding[cls][1][0]
                ]

        return y_img

    def load_y(self, label_size):
        """
        This function returns the ground truth label_map/segmentation mask
        with the class id on each pixel position
        """
        if self.label_file is None:
            print("WARNING: Label File Does not Exists...")
            return np.zeros((label_size[0], label_size[1]))

        y_img = np.zeros((label_size[0], label_size[1]))
        lbm = cv2.imread(self.label_file)
        lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
        for idx, cls in enumerate(self.color_coding):
            col_0 = np.zeros((label_size[0], label_size[1]))
            col_1 = np.zeros((label_size[0], label_size[1]))
            col_2 = np.zeros((label_size[0], label_size[1]))

            col_0[lbm[:, :, 0] == self.color_coding[cls][0][2]] = 1
            col_1[lbm[:, :, 1] == self.color_coding[cls][0][1]] = 1
            col_2[lbm[:, :, 2] == self.color_coding[cls][0][0]] = 1
            col = col_0 + col_1 + col_2
            if cls == "unlabeled":
                y_img[col == 3] = -1
            else:
                y_img[col == 3] = idx + 1

        return y_img

    def write_result(self, res_path, color_map):
        """
        This function save the result of the segmentation comparing it to the
        ground truth segmentation mask
        """
        im_id = os.path.basename(self.image_file)
        color_map = np.copy(color_map)
        height, width = color_map.shape[:2]
        label = self.load_y_as_color_map((height, width))
        if "unlabeled" in self.color_coding:
            col_0 = np.zeros((height, width))
            col_1 = np.zeros((height, width))
            col_2 = np.zeros((height, width))

            col_0[label[:, :, 0] == self.color_coding["unlabeled"][1][2]] = 1
            col_1[label[:, :, 1] == self.color_coding["unlabeled"][1][1]] = 1
            col_2[label[:, :, 2] == self.color_coding["unlabeled"][1][0]] = 1
            col = col_0 + col_1 + col_2
            index_y, index_x = np.where(col == 3)
            color_map[index_y, index_x, :] = [
                self.color_coding["unlabeled"][1][2],
                self.color_coding["unlabeled"][1][1],
                self.color_coding["unlabeled"][1][0]
            ]
        border = 255 * np.ones((height, 10, 3))
        result = np.concatenate([label, border, color_map], axis=1)
        res_file = os.path.join(res_path, im_id[:-4] + ".png")
        cv2.imwrite(res_file, result)

    def eval(self, color_map, stats_handler):
        """
        This function compares the ground truth segmentation mask
        with the prediction and stored it in the statistics modul
        """
        color_map = np.copy(color_map)
        height, width = color_map.shape[:2]
        if self.label_file is not None:
            lbm = self.load_y_as_color_map((height, width))
            if "unlabeled" in self.color_coding:
                col_0 = np.zeros((height, width))
                col_1 = np.zeros((height, width))
                col_2 = np.zeros((height, width))

                col_0[lbm[:, :, 0] == self.color_coding["unlabeled"][1][2]] = 1
                col_1[lbm[:, :, 1] == self.color_coding["unlabeled"][1][1]] = 1
                col_2[lbm[:, :, 2] == self.color_coding["unlabeled"][1][0]] = 1
                col = col_0 + col_1 + col_2
                index_y, index_x = np.where(col == 3)
                color_map[index_y, index_x, :] = [
                    self.color_coding["unlabeled"][1][2],
                    self.color_coding["unlabeled"][1][1],
                    self.color_coding["unlabeled"][1][0]
                ]

            for _, cls in enumerate(self.color_coding):
                if cls == "unlabeled":
                    continue
                cls_key = self.color_coding[cls][1]
                col_00 = np.zeros((height, width))
                col_01 = np.zeros((height, width))
                col_02 = np.zeros((height, width))
                col_00[lbm[:, :, 0] == cls_key[2]] = 1
                col_01[lbm[:, :, 1] == cls_key[1]] = 1
                col_02[lbm[:, :, 2] == cls_key[0]] = 1

                col_10 = np.zeros((height, width))
                col_11 = np.zeros((height, width))
                col_12 = np.zeros((height, width))
                col_10[color_map[:, :, 0] == cls_key[2]] = 1
                col_11[color_map[:, :, 1] == cls_key[1]] = 1
                col_12[color_map[:, :, 2] == cls_key[0]] = 1

                col_0 = col_00 + col_01 + col_02
                col_1 = col_10 + col_11 + col_12

                tp_map = np.zeros((height, width))
                fp_map = np.zeros((height, width))
                fn_map = np.zeros((height, width))

                tp_map[np.logical_and(col_0 == 3, col_1 == 3)] = 1
                fp_map[np.logical_and(col_0 != 3, col_1 == 3)] = 1
                fn_map[np.logical_and(col_0 == 3, col_1 != 3)] = 1

                stats_handler.count(cls, "tp", np.sum(tp_map))
                stats_handler.count(cls, "fp", np.sum(fp_map))
                stats_handler.count(cls, "fn", np.sum(fn_map))

    def visualize_result(self, vis_path, color_map):
        """
        This function saves the original image overlayed with the
        segmentation mask
        """
        im_id = os.path.basename(self.image_file)
        vis_file = os.path.join(vis_path, im_id)
        img_cont = ImageContainer(self.load_x())
        cv2.imwrite(vis_file, img_cont.overlay(color_map))
