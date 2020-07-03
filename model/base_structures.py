import numpy as np

from structured_classifier.decision_layer import DecisionLayer
from structured_classifier.decision_3d_layer import Decision3DLayer

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
        kernel=None,
        kernel_shape=None,
        ):
    if type(decision_type) is list:
        dt = np.random.choice(decision_type)
    else:
        dt = decision_type

    if dt == "kernel":
        xx = DecisionLayer(INPUTS=INPUTS,
                           name=name,
                           kernel=kernel,
                           kernel_shape=kernel_shape,
                           down_scale=down_scale,
                           clf=clf,
                           clf_options=clf_options,
                           data_reduction=data_reduction)
    elif dt in ["slic", "watershed", "felzenszwalb", "quickshift"]:
        xx = SuperPixelLayer(INPUTS=INPUTS,
                             name=name,
                             super_pixel_method=dt,
                             down_scale=down_scale,
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
            kernel=None,
            kernel_shape=None,
            ):

    if type(decision_type) is list:
        dt = np.random.choice(decision_type)
    else:
        dt = decision_type

    if dt == "kernel":
        xx = Decision3DLayer(
            INPUTS=INPUTS,
            name=name,
            kernel=kernel, kernel_shape=kernel_shape,
            down_scale=down_scale, data_reduction=data_reduction,
            clf=clf, clf_options=clf_options
        )
    elif dt in ["slic", "watershed", "felzenszwalb", "quickshift"]:
        k_t, k_x, k_y = kernel
        xx = SuperPixel3DLayer(
            INPUTS=INPUTS,
            name=name,
            super_pixel_method=dt,
            down_scale=down_scale, data_reduction=data_reduction,
            time_range=k_t,
            clf=clf, clf_options=clf_options
        )
    else:
        raise ValueError("Unknown decision_type: {}".format(dt))
    return xx