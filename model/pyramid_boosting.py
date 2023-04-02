from structured_classifier.input_layer import InputLayer
from structured_classifier.pixel_layer import PixelLayer
from structured_classifier.normalization_layer import NormalizationLayer

from model.model_blue_print import ModelBluePrint
from model.base_structures import get_decision_layer
from structured_classifier.model import Model


class PyramidBoosting(ModelBluePrint):
    def __init__(self,
                 image_width=None,
                 image_height=None,
                 initial_image_down_scale=None,
                 max_depth=3,
                 max_kernel_sum=5,
                 features_to_use="gray-lm",
                 kernel_shape="ellipse",
                 norm_input=None,
                 clf="mlp_x",
                 clf_options=None,
                 data_reduction=3):
        self.max_depth = max_depth
        self.max_kernel_sum = max_kernel_sum
        self.features_to_use = features_to_use
        self.norm_input = norm_input
        self.kernel_shape = kernel_shape

        self.clf = clf
        self.clf_options = clf_options
        self.data_reduction = data_reduction
        super(PyramidBoosting, self).__init__()
        self.model = self.build(
            width=image_width,
            height=image_height,
            initial_down_scale=initial_image_down_scale
        )

    def build(self, width=None, height=None, initial_down_scale=None):
        xx = InputLayer(name="in",
                        features_to_use=self.features_to_use,
                        initial_down_scale=initial_down_scale,
                        width=width,
                        height=height)

        if self.norm_input is not None:
            xx = NormalizationLayer(
                INPUTS=xx,
                name="norm_in",
                norm_option=self.norm_input
            )

        for d in range(self.max_depth):
            xx = PixelLayer(
                INPUTS=xx,
                name="estimator_stage_{}".format(self.max_depth - d - 1),
                kernel=(self.max_kernel_sum, self.max_kernel_sum),
                kernel_shape=self.kernel_shape,
                strides=(1, 1),
                down_scale=self.max_depth - d - 1,
                clf=self.clf,
                clf_options=self.clf_options,
                data_reduction=self.data_reduction
            )
        return Model(graph=xx)
