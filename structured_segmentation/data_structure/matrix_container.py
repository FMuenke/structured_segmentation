"""
This module contains the matrix container class, which is able to handle
matrices in general and read, merge and normalize values in it
"""

import numpy as np
import cv2
from skimage.measure import block_reduce


class MatrixContainer:
    """
    This class support various operations on matrices
    """
    def __init__(self, matrix):
        """
        Initialize this Object with the matrix o operate on
        """
        self.matrix = matrix

    def normalize(self):
        """
        Normalize the Matrix by mean and standard deviation
        """
        epsilon = 1e-6
        mean_mat = np.mean(self.matrix)
        var_mat = np.var(self.matrix)
        if var_mat != 0:
            mat_norm = (self.matrix - mean_mat) / var_mat
            min_mat = np.min(mat_norm)
            max_mat = np.max(mat_norm)
            mat_norm = (mat_norm - min_mat) / (max_mat - min_mat + epsilon)
        else:
            mat_norm = np.zeros(self.matrix.shape)
        return mat_norm

    def global_pooling(self, pooling_mode):
        """
        Reduce the whole matrix to a a vector representing the matrix
        pooling_mode: Mode how to summarize the matrix [max, min, mean, sum]
        """
        if pooling_mode == "max":
            vec = np.max(self.matrix, axis=0)
            vec = np.max(vec, axis=0)
        elif pooling_mode == "mean":
            vec = np.mean(self.matrix, axis=0)
            vec = np.mean(vec, axis=0)
        elif pooling_mode == "min":
            vec = np.min(self.matrix, axis=0)
            vec = np.min(vec, axis=0)
        elif pooling_mode == "sum":
            vec = np.sum(self.matrix, axis=0)
            vec = np.sum(vec, axis=0)
        else:
            raise ValueError("Error: Down-Sampling only supports none, min, max and mean")
        return np.reshape(vec, (1, -1))

    def pooling(self, down_sampling_mode, down_sampling_size):
        """
        Reduce the size of the matrix by summarizing cells with the define mode
        This works same as average_pooling in tensorflow
        down_sampling_mode: [max, mean, min]
        down_sampling_size: Size of the cells to be summarized
        """
        if down_sampling_mode == "max":
            mat = block_reduce(self.matrix, (down_sampling_size, down_sampling_size), np.max)
        elif down_sampling_mode == "mean":
            mat = block_reduce(self.matrix, (down_sampling_size, down_sampling_size), np.mean)
        elif down_sampling_mode == "min":
            mat = block_reduce(self.matrix, (down_sampling_size, down_sampling_size), np.min)
        else:
            raise ValueError("Error: Down-Sampling only supports none, min, max and mean")
        return mat

    def apply_convolution(self, kernel):
        """
        This function applies the defined kernel as convolution to the contained matrix
        kernel: numpy kernel to be applied
        """
        mat = np.copy(self.matrix)
        mat = mat.astype("float")
        mat = cv2.filter2D(mat, -1, kernel)
        return mat

    def cut_roi(self, center_point, roi_size):
        """
        This function cuts a region of interest from the matrix
        center_point: Center of Region of Interest (x, y)
        roi_size: Size of the Region of interest (int)
        """
        height, width = self.matrix.shape[:2]
        x_coord_1 = np.max([0, int(center_point[0] - roi_size / 2)])
        y_coord_1 = np.max([0, int(center_point[1] - roi_size / 2)])
        x_coord_2 = np.min([width, int(center_point[0] + roi_size / 2)])
        y_coord_2 = np.min([height, int(center_point[1] + roi_size / 2)])
        if len(self.matrix.shape) == 3:
            return self.matrix[y_coord_1:y_coord_2, x_coord_1:x_coord_2, :]
        return self.matrix[y_coord_1:y_coord_2, x_coord_1:x_coord_2]

    def merge(self, mode="max"):
        """
        Merges the Matrix Layers into on flat matrix
        mode: How to merge the matrix layers [max, min, mean]
        """
        if mode == "max":
            merged_f_map = np.max(self.matrix, axis=2)
        elif mode == "min":
            merged_f_map = np.min(self.matrix, axis=2)
            merged_f_map = np.multiply(merged_f_map, -1)
        elif mode == "mean":
            merged_f_map = np.mean(self.matrix, axis=2)
        else:
            raise ValueError("Error: mode option is not known!")
        return merged_f_map
