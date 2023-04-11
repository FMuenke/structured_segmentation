import numpy as np
import cv2

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.util import img_as_float

from layers.layer_operations import resize


def patches(img, patch_size):
    segment_id = 1
    height, width = img.shape[:2]
    segment_map = np.zeros((height, width))
    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            jmax = min(j + patch_size, height - 1)
            imax = min(i + patch_size, width - 1)
            segment_map[j:jmax, i:imax] = segment_id
            segment_id += 1

    return segment_map


def get_features_opt_quantiles(f_tensor, f_segments):
    x_img = []
    for u in np.unique(f_segments):
        ids = np.where(f_segments == u)[0]
        u_val = f_tensor[ids, :]
        x_seg = np.array([
            np.percentile(u_val, 0.95, axis=0),
            np.percentile(u_val, 0.85, axis=0),
            np.percentile(u_val, 0.75, axis=0),
            np.percentile(u_val, 0.65, axis=0),
            np.percentile(u_val, 0.55, axis=0),
            np.percentile(u_val, 0.45, axis=0),
            np.percentile(u_val, 0.35, axis=0),
            np.percentile(u_val, 0.25, axis=0),
            np.percentile(u_val, 0.15, axis=0),
            np.percentile(u_val, 0.05, axis=0),
        ])
        x_seg = np.reshape(x_seg, (1, -1))
        x_img.append(x_seg)
    return np.concatenate(x_img, axis=0)


def get_features_opt_sum(f_tensor, f_segments):
    x_img = []
    for u in np.unique(f_segments):
        ids = np.where(f_segments == u)[0]
        u_val = f_tensor[ids, :]
        x_seg = np.array([
            np.sum(u_val, axis=0),
        ])
        x_seg = np.reshape(x_seg, (1, -1))
        x_img.append(x_seg)
    return np.concatenate(x_img, axis=0)


def get_features_opt_gauss(f_tensor, f_segments):
    x_img = []
    for u in np.unique(f_segments):
        ids = np.where(f_segments == u)[0]
        u_val = f_tensor[ids, :]
        x_seg = np.array([
            np.mean(u_val, axis=0),
            np.std(u_val, axis=0)
        ])
        x_seg = np.reshape(x_seg, (1, -1))
        x_img.append(x_seg)
    return np.concatenate(x_img, axis=0)


def get_features_opt_histogram(f_tensor, f_segments, bins):
    x_img = []
    n_features = f_tensor.shape[1]
    m = np.max(f_tensor, axis=0)
    for u in np.unique(f_segments):
        ids = np.where(f_segments == u)[0]
        x_seg = []
        for i in range(n_features):
            m_i = np.max([m[i], 1])
            u_val = f_tensor[ids, i]
            u_x, _ = np.histogram(u_val, bins=bins, range=(0, m_i))
            x_seg.append(np.reshape(u_x, (1, -1)))
        x_img.append(np.concatenate(x_seg, axis=1))
    x_img = np.concatenate(x_img, axis=0)
    return x_img


def get_features_opt_hog(tensor, segments):
    def to_oriented_histograms(gx_, gy_):
        mag, ang = cv2.cartToPolar(gx_, gy_)
        ang_sorted = ang / (2 * np.pi) * orientations
        ang_sorted = ang_sorted.astype(np.int)

        gradient_maps = []
        for ori_idx in range(orientations):
            grad_map = np.zeros(mag.shape)
            grad_map[ang_sorted == ori_idx] = mag[ang_sorted == ori_idx]
            gradient_maps.append(grad_map)
        return np.stack(gradient_maps, axis=2)

    def norm_x_seg(x_seg_):
        if norm_option == "L2-HYS":
            x_seg_norm_ = np.clip(x_seg_ / np.sqrt((x_seg_ ** 2).sum() + eps), 0, 0.2)
            x_seg_norm_ /= np.sqrt((x_seg_norm_**2).sum() + eps)
            return x_seg_norm_
        if norm_option == "L2":
            x_seg_ /= np.sqrt((x_seg_**2).sum() + eps)
            return x_seg_
        return x_seg_

    x_img = []
    orientations = 9
    eps = 1e-6
    norm_option = "L2-HYS"
    h, w, num_f = tensor.shape
    f_segments = np.reshape(segments, (w * h))
    for i in range(0, num_f, 2):
        gx, gy = tensor[:, :, i], tensor[:, :, i+1]
        hog = to_oriented_histograms(gx, gy)

        f_hog = np.reshape(hog, (w * h, orientations))
        x_hog = []
        for u in np.unique(f_segments):
            ids = np.where(f_segments == u)[0]
            u_val = f_hog[ids, :]
            x_seg = np.sum(u_val, axis=0)
            x_seg = norm_x_seg(x_seg)
            x_seg = np.reshape(x_seg, (1, -1))
            x_hog.append(x_seg)
        x_hog = np.concatenate(x_hog, axis=0)
        x_img.append(x_hog)
    x_img = np.concatenate(x_img, axis=1)
    return x_img


def get_features_for_segments(tensor, segments, feature_aggregation):
    if len(tensor.shape) < 3:
        tensor = np.expand_dims(tensor, axis=2)
    h, w, num_f = tensor.shape
    segments = resize(segments, width=w, height=h)
    f_tensor = np.reshape(tensor, (w * h, num_f))
    f_segments = np.reshape(segments, (w * h))

    if "quantiles" == feature_aggregation:
        return get_features_opt_quantiles(f_tensor, f_segments)
    if "hist" in feature_aggregation:
        bins = int(feature_aggregation.replace("hist", ""))
        return get_features_opt_histogram(f_tensor, f_segments, bins)
    if "sum" == feature_aggregation:
        return get_features_opt_sum(f_tensor, f_segments)
    if "gauss" == feature_aggregation:
        return get_features_opt_gauss(f_tensor, f_segments)
    if "hog" == feature_aggregation:
        return get_features_opt_hog(tensor, segments)
    raise ValueError("FeatureAggregation - {} - not known!".format(feature_aggregation))


def generate_segments(x_input, opt):
    img = img_as_float(x_input[::2, ::2])
    height, width = img.shape[:2]
    patch_size = 8 * 2 ** 0
    n_segments = int(height / patch_size) * int(width / patch_size)
    if "felzenszwalb" in opt["super_pixel_method"]:
        scale = 2 ** opt["down_scale"]
        segments = felzenszwalb(img, scale=scale, sigma=0.5, min_size=50)
    elif "slic" in opt["super_pixel_method"]:
        segments = slic(img, n_segments=n_segments, compactness=10, sigma=1, start_label=1)
    elif "quickshift" in opt["super_pixel_method"]:
        kernel_size = 3 * (opt["down_scale"] + 1)
        segments = quickshift(img, kernel_size=kernel_size, max_dist=6, ratio=0.5)
    elif "watershed" in opt["super_pixel_method"]:
        gradient = sobel(rgb2gray(img))
        segments = watershed(gradient, markers=n_segments, compactness=0.001)
    elif "patches" in opt["super_pixel_method"]:
        segments = patches(img, patch_size=patch_size)
    else:
        raise ValueError("SuperPixel-Option: {} unknown!".format(opt["super_pixel_method"]))
    return segments


def map_segments(segments, y_pred):
    if len(y_pred.shape) == 1:
        y_pred = np.expand_dims(y_pred, axis=1)
    n_cls = y_pred.shape[1]
    label_map = np.zeros((segments.shape[0], segments.shape[1], n_cls))

    for p in range(n_cls):
        for i, u in enumerate(np.unique(segments)):
            iy, ix = np.where(segments == u)
            label_map[iy, ix, p] = y_pred[i, p]
    return label_map


def get_y_for_segments(y_img, segments):
    y = []
    segments = resize(segments, width=y_img.shape[1], height=y_img.shape[0])
    for u in np.unique(segments):
        counts = np.bincount(y_img[segments == u].astype(np.int))
        y_seg = np.argmax(counts)
        y.append(y_seg)
    return np.array(y)