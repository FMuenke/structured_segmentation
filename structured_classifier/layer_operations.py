import cv2
import numpy as np


def resize(data, width, height, interpolation="nearest"):
    if interpolation == "nearest":
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_NEAREST)
    elif interpolation == "linear":
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_LINEAR)
    elif interpolation == "cubic":
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError("Does not know interpolation: {}".format(interpolation))
    return data


def normalize(data):
    epsilon = 1e-6
    # mat_mean = np.mean(np.mean(data, axis=0), axis=0)
    mat_norm = data#  - mat_mean
    min_mat = np.min(np.min(mat_norm, axis=0), axis=0)
    max_mat = np.max(np.max(mat_norm, axis=0), axis=0)
    mat_norm = mat_norm / (max_mat - min_mat + epsilon)
    return mat_norm
