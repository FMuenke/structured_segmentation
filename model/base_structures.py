import numpy as np

from structured_classifier.pixel_layer import PixelLayer
from structured_classifier.graph_3d_layer import Graph3DLayer

from structured_classifier.super_pixel_layer import SuperPixelLayer
from structured_classifier.super_pixel_3d_layer import SuperPixel3DLayer


def get_decision_layer(
        INPUTS,
        name,
        decision_type,
        clf,
        clf_options,
        down_scale,
        data_reduction,
        feature_aggregation="hist32",
        kernel=None,
        strides=None,
        kernel_shape=None,
        ):
    if type(decision_type) is list:
        dt = np.random.choice(decision_type)
    else:
        dt = decision_type

    if dt == "kernel":
        xx = PixelLayer(INPUTS=INPUTS,
                        name=name,
                        kernel=kernel,
                        strides=strides,
                        kernel_shape=kernel_shape,
                        down_scale=down_scale,
                        clf=clf,
                        clf_options=clf_options,
                        data_reduction=data_reduction)
    elif dt in ["slic", "watershed", "felzenszwalb", "quickshift", "patches"]:
        xx = SuperPixelLayer(INPUTS=INPUTS,
                             name=name,
                             super_pixel_method=dt,
                             down_scale=down_scale,
                             feature_aggregation=feature_aggregation,
                             clf=clf,
                             clf_options=clf_options,
                             data_reduction=data_reduction)
    else:
        raise ValueError("Unknown decision_type: {}".format(dt))
    return xx


def get_decision_layer_3d(
            INPUTS,
            name,
            decision_type,
            clf,
            clf_options,
            down_scale,
            data_reduction,
            feature_aggregation="hist32",
            kernel=None,
            kernel_shape=None,
            ):

    if type(decision_type) is list:
        dt = np.random.choice(decision_type)
    else:
        dt = decision_type

    if dt == "kernel":
        xx = Graph3DLayer(
            INPUTS=INPUTS,
            name=name,
            kernel=kernel, kernel_shape=kernel_shape,
            down_scale=down_scale, data_reduction=data_reduction,
            clf=clf, clf_options=clf_options
        )
    elif dt in ["slic", "watershed", "felzenszwalb", "quickshift", "patches"]:
        k_t, k_x, k_y = kernel
        xx = SuperPixel3DLayer(
            INPUTS=INPUTS,
            name=name,
            super_pixel_method=dt,
            down_scale=down_scale, data_reduction=data_reduction,
            feature_aggregation=feature_aggregation,
            time_range=k_t,
            clf=clf, clf_options=clf_options
        )
    else:
        raise ValueError("Unknown decision_type: {}".format(dt))
    return xx
