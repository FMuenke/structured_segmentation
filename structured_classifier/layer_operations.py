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


def normalize_and_standardize(data):
    epsilon = 1e-6
    if len(data.shape) < 3:
        data = np.expand_dims(data, axis=2)
    num_f = data.shape[2]

    mat_mean = np.mean(np.mean(data, axis=0), axis=0)
    mat_std = np.std(np.std(data, axis=0), axis=0)
    mat_norm = data - mat_mean
    for i in range(num_f):
        if mat_std[i] != 0:
            mat_norm[:, :, i] = np.divide(mat_norm[:, :, i], mat_std[i])
        else:
            mat_norm[:, :, i] = 0

    min_mat = np.min(np.min(mat_norm, axis=0), axis=0)
    max_mat = np.max(np.max(mat_norm, axis=0), axis=0)
    mat_norm = mat_norm / (max_mat - min_mat + epsilon)
    return mat_norm
