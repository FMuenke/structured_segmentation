"""
This modul contains all code to build the encoder decoder model
"""

from layers import InputLayer
from layers import NormalizationLayer
from layers import StructuredClassifierLayer
from model.model import Model

from model.graph import Graph


class EncoderDecoder(Model):
    """
    The Encoder Decoder Model uses by default 6 structured classifier
    """
    def __init__(self,
                 image_width=None,
                 image_height=None,
                 initial_image_down_scale=None,
                 depth=3,
                 kernel_size=5,
                 stride_size=2,
                 features_to_use="gray-color",
                 norm_input=None,
                 coder_type="kernel",
                 kernel_shape="ellipse",
                 clf="extra_tree",
                 clf_options=None,
                 data_reduction=0.66):
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.features_to_use = features_to_use
        self.norm_input = norm_input
        self.coder_type = coder_type
        self.kernel_shape = kernel_shape

        self.clf = clf
        self.clf_options = clf_options
        self.data_reduction = data_reduction
        super(EncoderDecoder, self).__init__()
        self.model = self.build(
            width=image_width,
            height=image_height,
            initial_down_scale=initial_image_down_scale
        )

    def build(self, width=None, height=None, initial_down_scale=None):
        x_layer = InputLayer(
            name="in",
            features_to_use=self.features_to_use,
            initial_down_scale=initial_down_scale,
            width=width,
            height=height
        )

        if self.norm_input is not None:
            x_layer = NormalizationLayer(
                INPUTS=x_layer,
                name="norm_in",
                norm_option=self.norm_input
            )

        for d in range(self.depth):
            x_layer = StructuredClassifierLayer(
                INPUTS=x_layer,
                name="enc_{}".format(d),
                kernel=(self.kernel_size, self.kernel_size),
                strides=(self.stride_size, self.stride_size),
                kernel_shape=self.kernel_shape,
                down_scale=(d + 1),
                clf=self.clf, clf_options=self.clf_options,
                data_reduction=self.data_reduction
            )

        for d in range(self.depth):
            x_layer = StructuredClassifierLayer(
                INPUTS=x_layer,
                name="dec_{}".format(d),
                kernel=(self.kernel_size, self.kernel_size),
                strides=(self.stride_size, self.stride_size),
                kernel_shape=self.kernel_shape,
                down_scale=(self.depth - d - 1),
                clf=self.clf, clf_options=self.clf_options,
                data_reduction=self.data_reduction
            )
        return Graph(layer_stack=x_layer)
