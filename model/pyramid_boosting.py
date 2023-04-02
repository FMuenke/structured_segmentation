from layers import InputLayer
from layers import PixelLayer
from layers import NormalizationLayer

from model.base_model import BaseModel
from layers.model import Model


class PyramidBoosting(BaseModel):
    def __init__(self,
                 image_width=None,
                 image_height=None,
                 initial_image_down_scale=None,
                 max_depth=3,
                 kernel=5,
                 stride=1,
                 features_to_use="gray-lm",
                 kernel_shape="ellipse",
                 norm_input=None,
                 clf="extra_tree",
                 clf_options=None,
                 data_reduction=0.33):
        self.max_depth = max_depth
        self.kernel = kernel
        self.stride = stride
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
                name="stage_{}".format(self.max_depth - d - 1),
                kernel=(self.kernel, self.kernel),
                kernel_shape=self.kernel_shape,
                strides=(self.stride, self.stride),
                down_scale=self.max_depth - d - 1,
                clf=self.clf,
                clf_options=self.clf_options,
                data_reduction=self.data_reduction
            )
        return Model(graph=xx)
