from model.graph import Graph
from layers import InputLayer, PixelLayer
from model.model import Model


class PixelSegmentor(Model):
    def __init__(self,
                 image_width=None,
                 image_height=None,
                 initial_image_down_scale=None,
                 feature_to_use="gray-color",
                 clf="extra_tree",
                 kernel=5,
                 stride=2,
                 kernel_shape="ellipse",
                 data_reduction=0,
                 ):
        self.feature_to_use = feature_to_use
        self.kernel = kernel
        self.stride = stride
        self.kernel_shape = kernel_shape
        self.clf = clf
        self.data_reduction = data_reduction
        super(PixelSegmentor, self).__init__()
        self.model = self.build(
            width=image_width,
            height=image_height,
            initial_down_scale=initial_image_down_scale
        )

    def build(self, width, height, initial_down_scale):
        x = InputLayer(
            name="INPUT",
            features_to_use=self.feature_to_use,
            width=width,
            height=height,
            initial_down_scale=initial_down_scale
        )
        x = PixelLayer(
            INPUTS=x,
            name="DecisionLayer",
            kernel=(self.kernel, self.kernel),
            strides=(self.stride, self.stride),
            kernel_shape=self.kernel_shape,
            clf=self.clf,
            data_reduction=self.data_reduction,
        )
        return Graph(layer_stack=x)
