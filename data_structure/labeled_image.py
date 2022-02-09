import os
import numpy as np
from PIL import Image
import cv2
from data_structure.image_container import ImageContainer


def get_file_name(base_path, data_id, extensions):
    for ext in extensions:
        filename = os.path.join(base_path, data_id + ext)
        if os.path.isfile(filename):
            return filename
    return None


class LabeledImage:

    image_extensions = [".jpg", ".JPG", ".png", "PNG", ".jpeg", ".ppm"]
    label_extensions = [".png", ".tif", "_label.tif", "_label.tiff", ".tiff", ".ppm"]

    def __init__(self, base_path, data_id, color_coding):
        self.id = data_id
        self.path = base_path
        self.color_coding = color_coding

        self.image_path = os.path.join(base_path, "images")
        self.label_path = os.path.join(base_path, "labels")

        self.image_file = get_file_name(self.image_path, self.id, self.image_extensions)
        self.label_file = get_file_name(self.label_path, self.id, self.label_extensions)

    def summary(self):
        y = self.load_y([100, 100])
        unique, counts = np.unique(y, return_counts=True)
        return unique, counts

    def get_image_size(self):
        if self.image_file is None:
            raise Exception("NO IMAGE FILE AVAILABLE")
        im = Image.open(self.image_file)
        width, height = im.size
        return height, width

    def load_x(self):
        img = cv2.imread(self.image_file)
        if img is None:
            print(self.image_file)
        return img

    def load_y_as_color_map(self, label_size):
        y_img = np.zeros((label_size[0], label_size[1], 3))
        if self.label_file is not None:
            lbm = cv2.imread(self.label_file)
            lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                c0 = np.zeros((label_size[0], label_size[1]))
                c1 = np.zeros((label_size[0], label_size[1]))
                c2 = np.zeros((label_size[0], label_size[1]))

                c0[lbm[:, :, 0] == self.color_coding[cls][0][2]] = 1
                c1[lbm[:, :, 1] == self.color_coding[cls][0][1]] = 1
                c2[lbm[:, :, 2] == self.color_coding[cls][0][0]] = 1
                c = c0 + c1 + c2
                iy, ix = np.where(c == 3)
                y_img[iy, ix, :] = [self.color_coding[cls][1][2],
                                    self.color_coding[cls][1][1],
                                    self.color_coding[cls][1][0]]

        return y_img

    def load_y(self, label_size):
        y_img = np.zeros((label_size[0], label_size[1]))
        if self.label_file is not None:
            lbm = cv2.imread(self.label_file)
            lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                c0 = np.zeros((label_size[0], label_size[1]))
                c1 = np.zeros((label_size[0], label_size[1]))
                c2 = np.zeros((label_size[0], label_size[1]))

                c0[lbm[:, :, 0] == self.color_coding[cls][0][2]] = 1
                c1[lbm[:, :, 1] == self.color_coding[cls][0][1]] = 1
                c2[lbm[:, :, 2] == self.color_coding[cls][0][0]] = 1
                c = c0 + c1 + c2
                y_img[c == 3] = idx + 1

        return y_img

    def write_result(self, res_path, color_map):
        im_id = os.path.basename(self.image_file)
        h, w = color_map.shape[:2]
        label = self.load_y_as_color_map((h, w))
        border = 255 * np.ones((h, 10, 3))
        r = np.concatenate([label, border, color_map], axis=1)
        res_file = os.path.join(res_path, im_id[:-4] + ".png")
        cv2.imwrite(res_file, r)

    def eval(self, color_map, stats_handler):
        height, width = color_map.shape[:2]
        if self.label_file is not None:
            lbm = self.load_y_as_color_map((height, width))
            for idx, cls in enumerate(self.color_coding):
                cls_key = self.color_coding[cls][1]
                c00 = np.zeros((height, width))
                c01 = np.zeros((height, width))
                c02 = np.zeros((height, width))
                c00[lbm[:, :, 0] == cls_key[2]] = 1
                c01[lbm[:, :, 1] == cls_key[1]] = 1
                c02[lbm[:, :, 2] == cls_key[0]] = 1

                c10 = np.zeros((height, width))
                c11 = np.zeros((height, width))
                c12 = np.zeros((height, width))
                c10[color_map[:, :, 0] == cls_key[2]] = 1
                c11[color_map[:, :, 1] == cls_key[1]] = 1
                c12[color_map[:, :, 2] == cls_key[0]] = 1

                c0 = c00 + c01 + c02
                c1 = c10 + c11 + c12

                tp_map = np.zeros((height, width))
                fp_map = np.zeros((height, width))
                fn_map = np.zeros((height, width))

                tp_map[np.logical_and(c0 == 3, c1 == 3)] = 1
                fp_map[np.logical_and(c0 != 3, c1 == 3)] = 1
                fn_map[np.logical_and(c0 == 3, c1 != 3)] = 1

                stats_handler.count(cls, "tp", np.sum(tp_map))
                stats_handler.count(cls, "fp", np.sum(fp_map))
                stats_handler.count(cls, "fn", np.sum(fn_map))

    def visualize_result(self, vis_path, color_map):
        im_id = os.path.basename(self.image_file)
        vis_file = os.path.join(vis_path, im_id)
        img_h = ImageContainer(self.load_x())
        cv2.imwrite(vis_file, img_h.overlay(color_map))
