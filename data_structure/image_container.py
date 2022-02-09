import numpy as np
import cv2
from skimage.transform import integral_image

from data_structure.matrix_container import MatrixContainer


class ImageContainer:
    def __init__(self, image):
        self.image = image

    def prepare_image_for_processing(self, color_space):
        if color_space == "gray":
            return [self.gray()]
        elif color_space == "hsv":
            hsv = self.hsv()
            return [hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]
        elif color_space == "BGR":
            BGR = self.BGR()
            return [BGR[:, :, 0], BGR[:, :, 1], BGR[:, :, 2]]
        elif color_space == "RGB":
            RGB = self.RGB()
            return [RGB[:, :, 0], RGB[:, :, 1], RGB[:, :, 2]]
        elif color_space == "opponent":
            opponent = self.opponent()
            return [opponent[:, :, 0], opponent[:, :, 1], opponent[:, :, 2]]
        elif color_space == "rgb":
            rgb = self.rgb()
            return [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
        elif color_space in ["log_geo_mean", "lgm"]:
            lgm = self.log_geometric_mean_chromaticity()
            return [lgm[:, :, 0], lgm[:, :, 1], lgm[:, :, 2]]
        else:
            raise ValueError("Color Space {} unknown.".format_map(color_space))

    def log_geometric_mean_chromaticity(self):
        image = self.image.astype(np.int)
        B = np.add(image[:, :, 0], 1)
        G = np.add(image[:, :, 1], 1)
        R = np.add(image[:, :, 2], 1)
        geometric_mean = np.cbrt(np.multiply(np.multiply(B, G), R))
        bm = np.divide(B, geometric_mean)
        gm = np.divide(G, geometric_mean)
        rm = np.divide(R, geometric_mean)

        bm = np.log(bm)
        gm = np.log(gm)
        rm = np.log(rm)

        bm = np.expand_dims(bm, axis=2)
        gm = np.expand_dims(gm, axis=2)
        rm = np.expand_dims(rm, axis=2)
        return np.concatenate([rm, gm, bm], axis=2)

    def BGR(self):
        return self.image

    def RGB(self):
        B = self.image[:, :, 0]
        G = self.image[:, :, 1]
        R = self.image[:, :, 2]
        R = np.expand_dims(R, axis=2)
        B = np.expand_dims(B, axis=2)
        G = np.expand_dims(G, axis=2)
        return np.concatenate([R, G, B], axis=2)

    def gray(self):
        image = np.copy(self.image)
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    def integral(self):
        return integral_image(self.gray())

    def hsv(self):
        image = np.copy(self.image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def opponent(self):
        B = self.image[:, :, 0]
        G = self.image[:, :, 1]
        R = self.image[:, :, 2]
        O1 = np.divide((R - G), np.sqrt(2))
        O2 = np.divide((R + G - np.multiply(B, 2)), np.sqrt(6))
        O3 = np.divide((R + G + B), np.sqrt(3))
        O1 = np.expand_dims(O1, axis=2)
        O2 = np.expand_dims(O2, axis=2)
        O3 = np.expand_dims(O3, axis=2)
        return np.concatenate([O1, O2, O3], axis=2)

    def rgb(self):
        B = self.image[:, :, 0].astype(np.float)
        G = self.image[:, :, 1].astype(np.float)
        R = self.image[:, :, 2].astype(np.float)
        r = 100 * np.divide(R, (R + G + B + 1e-5))
        g = 100 * np.divide(G, (R + G + B + 1e-5))
        b = 100 * np.divide(B, (R + G + B + 1e-5))
        r = np.expand_dims(r, axis=2)
        b = np.expand_dims(b, axis=2)
        g = np.expand_dims(g, axis=2)
        return np.concatenate([r, b, g], axis=2)

    def normalize(self):
        image = np.copy(self.image)
        mat = MatrixContainer(image)
        img_norm = 255 * mat.normalize()
        return img_norm.astype(np.uint8)

    def resize(self, height, width):
        return cv2.resize(self.image,
                          (int(width), int(height)),
                          interpolation=cv2.INTER_CUBIC)

    def _scale_image_to_octave(self, octave):
        image = np.copy(self.image)
        height, width = image.shape[:2]
        if height < 2 or width < 2:
            return image
        elif octave == 0:
            return image
        else:
            oct_width = int(width / np.power(2, octave))
            oct_height = int(height / np.power(2, octave))
            return self.resize(height=oct_height, width=oct_width)

    def overlay(self, mask):
        height, width = self.image.shape[:2]
        mask = cv2.resize(mask, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        return cv2.addWeighted(self.image.astype(np.uint8), 0.5, mask.astype(np.uint8), 0.5, 0)
        # return mask
