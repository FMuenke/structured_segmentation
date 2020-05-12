import cv2
import numpy as np

from skimage.transform import rotate


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


def dropout(data, drop_rate):
    n_features = data.shape[1]
    for i in range(data.shape[0]):
        drop = np.random.choice(n_features, size=int(n_features*drop_rate))
        data[i, drop] = 0
    return data


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X


def augment_tag(data, label):
    ang = np.random.randint(0, 179)
    data = rotate(np.copy(data), angle=ang)
    label = rotate(np.copy(label), angle=ang)
    if np.random.randint(2):
        data = data[:, ::-1]
        label = label[:, ::-1]
    if np.random.randint(2):
        data = data[::-1, :]
        label = label[::-1, :]

    if np.random.randint(5):
        movement = 20
        height, width = data.shape[:2]
        dx = int(np.random.randint(movement) / 100 * width)
        dy = int(np.random.randint(movement) / 100 * height)
        data = shift_image(data, dx, dy)
    return data, label
