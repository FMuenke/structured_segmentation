import numpy as np
import cv2
from skimage.filters import frangi, hessian, sato

from structured_classifier.layer_operations import normalize, resize


class TopClippingPercentile:
    list_of_parameters = [
        None, 1, 5, 10, 25, 50
    ]
    key = "top_clipping_percentile"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = np.clip(x_img, 0, np.percentile(x_img, self.parameter))
        return x_img


class FrangiFilter:
    list_of_parameters = [
        None, 1
    ]
    key = "frangi"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        x_img = frangi(x_img)
        x_img = normalize(x_img)
        return x_img


class EdgeDetector:
    list_of_parameters = [
        None, 1, 2, 3, 4, 5, 6
    ]
    key = "edge"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        k = 2**self.parameter + 1
        x_img_blur = cv2.blur(np.copy(x_img.astype(np.uint8)), ksize=(k, k))
        ed_xy = np.abs(x_img.astype(np.float64) - x_img_blur.astype(np.float64))
        return normalize(ed_xy)


class Blurring:
    list_of_parameters = [
        None, 3, 5, 7, 9, 11, 13, 15
    ]
    key = "blurring"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        x_img = cv2.blur(np.copy(x_img.astype(np.uint8)), ksize=(self.parameter, self.parameter))
        return x_img.astype(np.float64) / 255


class Threshold:
    list_of_parameters = [
        None, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9
    ]
    key = "threshold"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img[x_img < self.parameter] = 0
        x_img[x_img >= self.parameter] = 1
        return x_img


class ThresholdPercentile:
    list_of_parameters = [
        None, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99
    ]
    key = "threshold_percentile"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        threshold = np.percentile(x_img, self.parameter)
        x_img[x_img < threshold] = 0
        x_img[x_img >= threshold] = 1
        return x_img


class MorphologicalOpening:
    list_of_parameters = [
        None, 3, 5, 7, 9, 11, 13, 15
    ]
    key = "opening"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        kernel = np.ones((self.parameter, self.parameter), np.uint8)
        x_img = 255 * x_img
        x_img = cv2.morphologyEx(x_img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return x_img.astype(np.float64) / 255


class MorphologicalErosion:
    list_of_parameters = [
        None, 3, 5, 7, 9, 11, 13, 15
    ]
    key = "erode"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        kernel = np.ones((self.parameter, self.parameter), np.uint8)
        x_img = 255 * x_img
        x_img = cv2.morphologyEx(x_img.astype(np.uint8), cv2.MORPH_ERODE, kernel)
        return x_img.astype(np.float64) / 255


class MorphologicalDilatation:
    list_of_parameters = [
        None, 3, 5, 7, 9, 11, 13, 15
    ]
    key = "dilate"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        kernel = np.ones((self.parameter, self.parameter), np.uint8)
        x_img = 255 * x_img
        x_img = cv2.morphologyEx(x_img.astype(np.uint8), cv2.MORPH_DILATE, kernel)
        return x_img.astype(np.float64) / 255


class MorphologicalClosing:
    list_of_parameters = [
        None, 3, 5, 7, 9, 11, 13, 15
    ]
    key = "closing"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        kernel = np.ones((self.parameter, self.parameter), np.uint8)
        x_img = 255 * x_img
        x_img = cv2.morphologyEx(x_img.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return x_img.astype(np.float64) / 255


class Invert:
    list_of_parameters = [
        -1, 1
    ]
    key = "invert"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter == -1:
            stamp = np.ones(x_img.shape) * np.max(x_img)
            return stamp - x_img
        else:
            return x_img


class Resize:
    list_of_parameters = [
        None, 32, 64, 128, 256, 512
    ]
    key = "resize"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        return resize(x_img, width=self.parameter, height=self.parameter)