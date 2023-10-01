"""
This modul handles the class image container used to easily handle images
"""

import numpy as np
import cv2
from skimage.transform import integral_image

from structured_segmentation.data_structure.matrix_container import MatrixContainer


class ImageContainer:
    """
    The image container class implements many methods to apply to images
    """
    def __init__(self, image):
        """
        Initialize the objet with the image numpy array
        """
        self.image = image

    def prepare_image_for_processing(self, color_space):
        """
        This function is called before feature extraction and transforms the image
        to the defined color-space. This already implements a part of the feature processing
        """
        if color_space == "gray":
            return [self.gray()]
        if color_space == "hsv":
            hsv = self.hsv()
            return [hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]
        if color_space == "RGB":
            re_gr_bl = self.red_green_blue()
            return [re_gr_bl[:, :, 0], re_gr_bl[:, :, 1], re_gr_bl[:, :, 2]]
        if color_space == "opponent":
            opponent = self.opponent()
            return [opponent[:, :, 0], opponent[:, :, 1], opponent[:, :, 2]]
        if color_space == "rgb":
            rgb = self.rgb()
            return [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
        if color_space in ["log_geo_mean", "lgm"]:
            lgm = self.log_geometric_mean_chromaticity()
            return [lgm[:, :, 0], lgm[:, :, 1], lgm[:, :, 2]]
        raise ValueError("Color Space {} unknown.".format_map(color_space))

    def log_geometric_mean_chromaticity(self):
        """
        This function returns the log_geometric_mean_chromaticity colorspace
        """
        image = self.image.astype(np.int)
        blue = np.add(image[:, :, 0], 1)
        green = np.add(image[:, :, 1], 1)
        red = np.add(image[:, :, 2], 1)
        geometric_mean = np.cbrt(np.multiply(np.multiply(blue, green), red))
        blue_m = np.divide(blue, geometric_mean)
        green_m = np.divide(green, geometric_mean)
        red_m = np.divide(red, geometric_mean)

        blue_m = np.log(blue_m)
        green_m = np.log(green_m)
        red_m = np.log(red_m)

        blue_m = np.expand_dims(blue_m, axis=2)
        green_m = np.expand_dims(green_m, axis=2)
        red_m = np.expand_dims(red_m, axis=2)
        return np.concatenate([red_m, green_m, blue_m], axis=2)

    def blue_green_red(self):
        """
        This function returns the image in the regular cv2 BGR image format
        """
        return self.image

    def red_green_blue(self):
        """
        This function returns the image in the channel-swapped RGB image format
        """
        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]
        red = np.expand_dims(red, axis=2)
        blue = np.expand_dims(blue, axis=2)
        green = np.expand_dims(green, axis=2)
        return np.concatenate([red, green, blue], axis=2)

    def gray(self):
        """
        This function returns the image in a gray-scale image format
        """
        image = np.copy(self.image)
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    def integral(self):
        """
        This function returns the integral of the image
        """
        return integral_image(self.gray())

    def hsv(self):
        """
        This function returns the image transformed to the HSV color-space
        H (Hue) S (Saturation) V (Value)
        """
        image = np.copy(self.image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def opponent(self):
        """
        This function returns the image in the opponent color-space
        """
        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]
        opp_1 = np.divide((red - green), np.sqrt(2))
        opp_2 = np.divide((red + green - np.multiply(blue, 2)), np.sqrt(6))
        opp_3 = np.divide((red + green + blue), np.sqrt(3))
        opp_1 = np.expand_dims(opp_1, axis=2)
        opp_2 = np.expand_dims(opp_2, axis=2)
        opp_3 = np.expand_dims(opp_3, axis=2)
        return np.concatenate([opp_1, opp_2, opp_3], axis=2)

    def rgb(self):
        """
        This function returns the image as NORMALIZED rgb image
        """
        blue = self.image[:, :, 0].astype(np.float)
        green = self.image[:, :, 1].astype(np.float)
        red = self.image[:, :, 2].astype(np.float)
        red_norm = 100 * np.divide(red, (red + green + blue + 1e-5))
        green_norm = 100 * np.divide(green, (red + green + blue + 1e-5))
        blue_norm = 100 * np.divide(blue, (red + green + blue + 1e-5))
        red_norm = np.expand_dims(red_norm, axis=2)
        blue_norm = np.expand_dims(blue_norm, axis=2)
        green_norm = np.expand_dims(green_norm, axis=2)
        return np.concatenate([red_norm, blue_norm, green_norm], axis=2)

    def normalize(self):
        """
        This function returns the image normalized by mean and standard deviation
        """
        image = np.copy(self.image)
        mat = MatrixContainer(image)
        img_norm = 255 * mat.normalize()
        return img_norm.astype(np.uint8)

    def resize(self, height, width):
        """
        This function returns the image resized by the given dimensions
        height: new height of the image (int)
        width: new width of the image (int)
        """
        return cv2.resize(
            self.image,
            (int(width), int(height)),
            interpolation=cv2.INTER_CUBIC
        )

    def _scale_image_to_octave(self, octave):
        """
        This function resizes the image with a given octave
        Where the octave equals to the power of two (np.power(2, octave)
        octave: Power of 2 to reduce the image size (int) [0, 1, 2, ...]
        """
        image = np.copy(self.image)
        height, width = image.shape[:2]
        if height < 2 or width < 2:
            return image
        if octave == 0:
            return image
        oct_width = int(width / np.power(2, octave))
        oct_height = int(height / np.power(2, octave))
        return self.resize(height=oct_height, width=oct_width)

    def overlay(self, mask):
        """
        This function returns the image ovelayed with the provided mask
        mask: np.array to overlay
        """
        height, width = self.image.shape[:2]
        mask = cv2.resize(mask, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        return cv2.addWeighted(self.image.astype(np.uint8), 0.5, mask.astype(np.uint8), 0.5, 0)
