from structured_classifier.input_layer import InputLayer
from structured_classifier.normalization_layer import NormalizationLayer
from structured_classifier.bottle_neck_layer import BottleNeckLayer
from structured_classifier.voting_layer import VotingLayer
from structured_classifier.pixel_layer import PixelLayer

from model.base_structures import get_decision_layer


class PatchWork:
    def __init__(self,
                 patch_types,
                 down_scales,
                 features,
                 norm_input=None,
                 clf="b_rf",
                 clf_options=None,
                 data_reduction=0
                 ):

        self.patch_types = patch_types
        self.down_scales = down_scales
        self.features = features
        self.norm_input = norm_input

        self.clf = clf
        self.clf_options = clf_options
        self.data_reduction = data_reduction

        self.width = None
        self.height = None
        self.initial_down_scale = None

    def _build_patch_unit(self, idx, decision_type, down_scale, feature_aggregation):
        feature, aggregation = feature_aggregation.split("+")
        xx = InputLayer(
            name="in_{}".format(idx),
            features_to_use=feature,
            initial_down_scale=self.initial_down_scale,
            width=self.width,
            height=self.height)

        if self.norm_input is not None:
            xx = NormalizationLayer(INPUTS=xx, name="norm_{}".format(idx), norm_option=self.norm_input)

        xx = get_decision_layer(
            INPUTS=xx,
            name="patch_{}".format(idx),
            decision_type=decision_type,
            clf=self.clf,
            clf_options=self.clf_options,
            down_scale=down_scale,
            data_reduction=self.data_reduction,
            feature_aggregation=aggregation,
        )
        return xx

    def build(self, width=None, height=None, initial_down_scale=None, output_option="voting"):
        self.width = width
        self.height = height
        self.initial_down_scale = initial_down_scale
        i = 0

        model = []
        for p_type in self.patch_types:
            for d in self.down_scales:
                for feature_aggregation in self.features:
                    xx = self._build_patch_unit(idx=i,
                                                decision_type=p_type,
                                                down_scale=d,
                                                feature_aggregation=feature_aggregation)
                    i += 1
                    model.append(xx)

        if output_option == "voting":
            return VotingLayer(INPUTS=model, name="voting")
        elif output_option == "boosting":
            b = BottleNeckLayer(INPUTS=model, name="cls_preparation")
            return PixelLayer(INPUTS=b, name="boosting", clf=self.clf, clf_options=self.clf_options, kernel=(5, 5))
        return model
