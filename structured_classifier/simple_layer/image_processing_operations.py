import numpy as np
import cv2
from skimage.filters import frangi, threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.feature import canny
from skimage.segmentation import watershed, random_walker
from structured_classifier.layer_operations import normalize, resize


class Watershed:
    list_of_parameters = [None, 1, 5, 10]
    key = "watershed"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        markers = np.zeros(x_img.shape, dtype=np.uint)
        markers[x_img <= np.percentile(x_img, self.parameter)] = 1
        markers[x_img >= np.percentile(x_img, 100 - self.parameter)] = 2
        if np.min(markers) == 1:
            return markers - 1

        labels = watershed(x_img, markers)
        return labels.astype(np.float64) - 1


class FillContours:
    list_of_parameters = [None, 1]
    key = "fill_contours"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img[x_img < 0.5] = 0
        x_img[x_img >= 0.5] = 1
        x_img = 255 * x_img
        cnt, _ = cv2.findContours(x_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x_img = cv2.fillPoly(x_img, pts=cnt, color=255)
        return x_img.astype(np.float64) / 255


class LocalNormalization:
    list_of_parameters = [None, 4+1, 8+1, 16+1, 32+1, 64+1, 128+1]
    key = "local_normalization"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        total_avg = np.mean(x_img)
        avg = cv2.filter2D(
            np.copy(x_img), -1,
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.parameter, self.parameter))
        )
        img_norm = x_img - total_avg + avg
        return img_norm.astype(np.float64) / 255


class RemoveSmallObjects:
    list_of_parameters = [None, 2, 4, 8, 16, 32, 64, 128, 256]
    key = "remove_small_objects"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = x_img.astype(np.int)
        if len(np.unique(x_img)) == 1:
            return x_img
        x_img = remove_small_objects(x_img.astype(np.bool), min_size=self.parameter)
        return x_img.astype(np.float64)


class RemoveSmallHoles:
    list_of_parameters = [None, 2, 4, 8, 16, 32, 64, 128, 256]
    key = "remove_small_holes"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = x_img.astype(np.int)
        if len(np.unique(x_img)) == 1:
            return x_img
        x_img = remove_small_holes(x_img.astype(np.bool), area_threshold=self.parameter)
        return x_img.astype(np.float64)


class CannyEdgeDetector:
    list_of_parameters = [None, 1, 3, 5, 9, 17, 33]
    key = "canny_edge"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        x_img = canny(x_img.astype(np.uint8), sigma=self.parameter)
        return x_img.astype(np.float64) / 255


class TopClippingPercentile:
    list_of_parameters = [None, 1, 5, 10, 25, 50]
    key = "top_clipping_percentile"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = np.clip(x_img, 0, np.percentile(x_img, 100 - self.parameter))
        return x_img


class BottomClippingPercentile:
    list_of_parameters = [None, 1, 5, 10, 25, 50]
    key = "bottom_clipping_percentile"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = np.clip(x_img, np.percentile(x_img, self.parameter), 1)
        return x_img


class FrangiFilter:
    list_of_parameters = [None, 1]
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
    list_of_parameters = [None, 2+1, 4+1, 8+1, 16+1]
    key = "edge"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        x_img = cv2.Laplacian(np.copy(x_img.astype(np.uint8)), -1, ksize=self.parameter)
        return normalize(x_img.astype(np.float64))


class Blurring:
    list_of_parameters = [None, 2+1, 4+1, 8+1, 16+1, 32+1, 64+1]
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
    list_of_parameters = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
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
    list_of_parameters = [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99]
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


class ThresholdOtsu:
    list_of_parameters = [None, 1]
    key = "threshold_otsu"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        threshold = threshold_otsu(x_img)
        x_img[x_img < threshold] = 0
        x_img[x_img >= threshold] = 1
        return x_img


class LocalThreshold:
    list_of_parameters = [None, 8+1, 16+1, 32+1, 64+1, 128+1, 256+1]
    key = "local_threshold"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        threshold = threshold_local(x_img, block_size=self.parameter)
        x_img[x_img < threshold] = 0
        x_img[x_img >= threshold] = 1
        return x_img


class MorphologicalOpening:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1, 32 + 1]
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


class NegativeMorphologicalOpening:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1, 32 + 1]
    key = "negative_opening"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        op = MorphologicalOpening(self.parameter)
        x_img_morph = op.inference(x_img)
        return np.abs(x_img - x_img_morph)


class MorphologicalErosion:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1, 32 + 1]
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


class NegativeMorphologicalErosion:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1, 32 + 1]
    key = "negative_erode"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        op = MorphologicalErosion(self.parameter)
        x_img_morph = op.inference(x_img)
        return np.abs(x_img - x_img_morph)


class MorphologicalDilatation:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1, 32 + 1]
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


class NegativeMorphologicalDilatation:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1, 32 + 1]
    key = "negative_dilate"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        op = MorphologicalDilatation(self.parameter)
        x_img_morph = op.inference(x_img)
        return np.abs(x_img - x_img_morph)


class MorphologicalClosing:
    list_of_parameters = [None, 2+1, 4+1, 8+1, 16+1, 32+1]
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


class NegativeMorphologicalClosing:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1, 32 + 1]
    key = "negative_closing"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        op = MorphologicalClosing(self.parameter)
        x_img_morph = op.inference(x_img)
        return np.abs(x_img - x_img_morph)


class Invert:
    list_of_parameters = [-1, 1]
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
    list_of_parameters = [None, 2, 4, 8, 16, 32]
    key = "resize"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        height, width = x_img.shape[:2]
        return resize(x_img, width=int(width / self.parameter), height=int(height / self.parameter))


LIST_OF_OPERATIONS = [
    Invert,
    MorphologicalClosing,
    EdgeDetector,
    FrangiFilter,
    Threshold,
    MorphologicalOpening,
    NegativeMorphologicalOpening,
    MorphologicalClosing,
    NegativeMorphologicalClosing,
    ThresholdPercentile,
    MorphologicalDilatation,
    NegativeMorphologicalDilatation,
    MorphologicalErosion,
    NegativeMorphologicalErosion,
    Blurring,
    TopClippingPercentile,
    CannyEdgeDetector,
    LocalNormalization,
    RemoveSmallObjects,
    RemoveSmallHoles,
    ThresholdOtsu,
    LocalThreshold,
    FillContours,
    Watershed,
]
