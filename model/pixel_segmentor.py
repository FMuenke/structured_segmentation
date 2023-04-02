from structured_classifier.model import Model
from structured_classifier import InputLayer, PixelLayer
from model.model_blue_print import ModelBluePrint


class PixelSegmentor(ModelBluePrint):
    def __init__(self,
                 image_width=None,
                 image_height=None,
                 initial_image_down_scale=None,
                 feature_to_use="hsv-lm",
                 clf="mlp_x",
                 kernel=(1, 1),
                 data_reduction=0,
                 ):
        self.feature_to_use = feature_to_use
        self.kernel = kernel
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
            width=width, height=height,
            initial_down_scale=initial_down_scale
        )
        x = PixelLayer(
            INPUTS=x,
            name="DecisionLayer",
            kernel=self.kernel,
            clf=self.clf,
            data_reduction=self.data_reduction,
        )
        return Model(graph=x)
