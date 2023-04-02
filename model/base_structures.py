import numpy as np

from structured_classifier.pixel_layer import PixelLayer

from structured_classifier.super_pixel_layer.super_pixel_layer import SuperPixelLayer


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
