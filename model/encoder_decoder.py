import numpy as np

from structured_classifier.decision_layer import DecisionLayer
from structured_classifier.super_pixel_layer import SuperPixelLayer
from structured_classifier.input_layer import InputLayer
from structured_classifier.normalization_layer import NormalizationLayer

from model.base_structures import get_decision_layer


class EncoderDecoder:
    def __init__(self,
                 depth=5,
                 repeat=1,
                 max_kernel_sum=5,
                 features_to_use="gray-color",
                 norm_input=None,
                 coder_type="kernel",
                 kernel_shape="square",
                 clf="b_rf",
                 clf_options=None,
                 data_reduction=3):
        self.repeat = repeat
        self.depth = depth
        self.max_kernel_sum = max_kernel_sum
        self.features_to_use = features_to_use
        self.norm_input = norm_input
        self.coder_type = coder_type
        self.kernel_shape = kernel_shape

        self.clf = clf
        self.clf_options = clf_options
        self.data_reduction = data_reduction

    def build(self, width=None, height=None, initial_down_scale=None):
        xx = InputLayer(name="in",
                        features_to_use=self.features_to_use,
                        initial_down_scale=initial_down_scale,
                        width=width,
                        height=height)

        kernel = (self.max_kernel_sum, self.max_kernel_sum)

        if self.norm_input is not None:
            xx = NormalizationLayer(INPUTS=xx,
                                    name="norm_in",
                                    norm_option=self.norm_input)

        for d in range(self.depth):
            for r in range(self.repeat):
                xx = get_decision_layer(INPUTS=xx,
                                        name="enc_{}_{}".format(d, r),
                                        decision_type=self.coder_type,
                                        kernel=kernel,
                                        kernel_shape=self.kernel_shape,
                                        down_scale=d,
                                        clf=self.clf, clf_options=self.clf_options,
                                        data_reduction=self.data_reduction)

        for r in range(self.repeat):
            xx = get_decision_layer(INPUTS=xx,
                                    name="lat_{}".format(r),
                                    decision_type=self.coder_type,
                                    kernel=kernel,
                                    kernel_shape=self.kernel_shape,
                                    down_scale=self.depth,
                                    clf=self.clf, clf_options=self.clf_options,
                                    data_reduction=self.data_reduction)

        for d in range(self.depth):
            for r in range(self.repeat):
                xx = get_decision_layer(INPUTS=xx,
                                        name="dec_{}_{}".format(self.depth - d - 1, r),
                                        decision_type=self.coder_type,
                                        kernel=kernel,
                                        kernel_shape=self.kernel_shape,
                                        down_scale=self.depth - d - 1,
                                        clf=self.clf, clf_options=self.clf_options,
                                        data_reduction=self.data_reduction)
        return xx
