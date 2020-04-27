import os
import cv2
import numpy as np

from data_structure.image_handler import ImageHandler


class LbmTag:
    def __init__(self, path_to_image_file, color_coding):
        self.path_to_image_file = path_to_image_file
        self.path_to_label_file = self.get_pot_label_path()

        self.color_coding = color_coding

    def summary(self):
        y = self.load_y([100, 100])
        unique, counts = np.unique(y, return_counts=True)
        return unique, counts

    def get_pot_label_path(self):
        """
        Used to guess the matching ground truth labelfile
        Args:
            img_id: complete path to image

        Returns:
            estimated full path to label file
        """
        path_to_label_file = self.path_to_image_file.replace("images", "labels")

        path_to_label_file = path_to_label_file[:-4] + ".png"
        if os.path.isfile(path_to_label_file):
            return path_to_label_file
        path_to_label_file = path_to_label_file[:-4] + ".tif"
        if os.path.isfile(path_to_label_file):
            return path_to_label_file
        path_to_label_file = path_to_label_file[:-4] + ".tiff"
        if os.path.isfile(path_to_label_file):
            return path_to_label_file
        path_to_label_file = path_to_label_file[:-5] + "_label.tiff"
        if os.path.isfile(path_to_label_file):
            return path_to_label_file
        path_to_label_file = path_to_label_file[:-5] + ".tif"
        if os.path.isfile(path_to_label_file):
            return path_to_label_file
        return None

    def load_x(self):
        img = cv2.imread(self.path_to_image_file)
        if img is None:
            print(self.path_to_image_file)
        return img

    def load_y_as_color_map(self, label_size):
        y_img = np.zeros((label_size[0], label_size[1], 3))
        if self.path_to_label_file is not None:
            lbm = cv2.imread(self.path_to_label_file)
            lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                for x in range(label_size[1]):
                    for y in range(label_size[0]):
                        if lbm[y, x, 0] == self.color_coding[cls][0][2] \
                                and lbm[y, x, 1] == self.color_coding[cls][0][1] \
                                and lbm[y, x, 2] == self.color_coding[cls][0][0]:
                            y_img[y, x, :] = self.color_coding[cls][1]

        return y_img

    def load_y(self, label_size):
        y_img = np.zeros((label_size[0], label_size[1]))
        if self.path_to_label_file is not None:
            lbm = cv2.imread(self.path_to_label_file)
            lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                for x in range(label_size[1]):
                    for y in range(label_size[0]):
                        if lbm[y, x, 0] == self.color_coding[cls][0][2] \
                                and lbm[y, x, 1] == self.color_coding[cls][0][1] \
                                and lbm[y, x, 2] == self.color_coding[cls][0][0]:
                            y_img[y, x] = idx + 1
        return y_img

    def write_result(self, res_path, color_map):
        im_id = os.path.basename(self.path_to_image_file)
        h, w = color_map.shape[:2]
        label = self.load_y_as_color_map((h, w))
        border = 255 * np.ones((h, 10, 3))
        r = np.concatenate([label, border, color_map], axis=1)
        res_file = os.path.join(res_path, im_id[:-4] + ".png")
        cv2.imwrite(res_file, r)

    def eval(self, color_map, stats_handler):
        height, width = color_map.shape[:2]
        if self.path_to_label_file is not None:
            lbm = cv2.imread(self.path_to_label_file)
            lbm = cv2.resize(lbm, (width, height), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                for x in range(width):
                    for y in range(height):
                        a = lbm[y, x, :] == self.color_coding[cls][0]
                        b = color_map[y, x, :] == self.color_coding[cls][1]
                        if a.all():
                            if b.all():
                                stats_handler.count(cls, "tp")
                            else:
                                stats_handler.count(cls, "fn")
                        else:
                            if b.all():
                                stats_handler.count(cls, "fp")

    def visualize_result(self, vis_path, color_map):
        im_id = os.path.basename(self.path_to_image_file)
        vis_file = os.path.join(vis_path, im_id)
        img_h = ImageHandler(self.load_x())
        cv2.imwrite(vis_file, img_h.overlay(color_map))
