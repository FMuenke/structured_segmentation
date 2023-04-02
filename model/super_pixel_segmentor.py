from layers.model import Model
from layers import InputLayer, SuperPixelLayer
from model.base_model import BaseModel


class SuperPixelSegmentor(BaseModel):
    def __init__(self,
                 image_width=None,
                 image_height=None,
                 initial_image_down_scale=0,
                 super_pixel_down_scale=0,
                 feature_to_use="hsv-lm",
                 feature_aggregation="gauss",
                 super_pixel_method="slic",
                 clf="extra_tree",
                 clf_options=None,
                 data_reduction=0,
                 ):
        self.feature_to_use = feature_to_use
        self.feature_aggregation = feature_aggregation
        self.super_pixel_method = super_pixel_method
        self.clf = clf
        self.clf_options = clf_options
        self.data_reduction = data_reduction
        super(SuperPixelSegmentor, self).__init__()
        self.model = self.build(
            width=image_width,
            height=image_height,
            initial_down_scale=initial_image_down_scale,
            super_pixel_down_scale=super_pixel_down_scale,
        )

    def build(self, width, height, initial_down_scale, super_pixel_down_scale):
        x = InputLayer(
            name="INPUT",
            features_to_use=self.feature_to_use,
            width=width, height=height,
            initial_down_scale=initial_down_scale
        )
        x = SuperPixelLayer(
            INPUTS=x,
            name="DecisionLayer",
            super_pixel_method=self.super_pixel_method,
            feature_aggregation=self.feature_aggregation,
            clf=self.clf,
            clf_options=self.clf_options,
            data_reduction=self.data_reduction,
            down_scale=super_pixel_down_scale,
        )
        return Model(graph=x)
