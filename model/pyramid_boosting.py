from structured_classifier.input_3d_layer import Input3DLayer
from structured_classifier.input_layer import InputLayer
from structured_classifier.normalization_layer import NormalizationLayer
from structured_classifier.bottle_neck_3d_layer import BottleNeck3DLayer

from model.base_structures import get_decision_layer, get_decision_layer_3d


class PyramidBoosting:
    def __init__(self,
                 n_estimators=3,
                 max_depth=5,
                 max_kernel_sum=5,
                 features_to_use="gray-color",
                 decision_type="kernel",
                 kernel_shape="square",
                 feature_aggregation="hist32",
                 norm_input=None,
                 clf="b_rf",
                 clf_options=None,
                 data_reduction=6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_kernel_sum = max_kernel_sum
        self.features_to_use = features_to_use
        self.norm_input = norm_input
        self.kernel_shape = kernel_shape
        self.decision_type = decision_type
        self.feature_aggregation = feature_aggregation

        self.clf = clf
        self.clf_options = clf_options
        self.data_reduction = data_reduction

    def build(self, width=None, height=None, initial_down_scale=None):
        xx = InputLayer(name="in",
                        features_to_use=self.features_to_use,
                        initial_down_scale=initial_down_scale,
                        width=width,
                        height=height)

        if self.norm_input is not None:
            xx = NormalizationLayer(INPUTS=xx,
                                    name="norm_in",
                                    norm_option=self.norm_input)

        for i in range(self.n_estimators):
            xx = self._build_single_unit(xx, name="estimator_{}".format(i))
        return xx

    def _build_single_unit(self, input_layer, name):
        x1 = input_layer

        kernel = (self.max_kernel_sum, self.max_kernel_sum)

        for d in range(self.max_depth):
            x1 = get_decision_layer(INPUTS=x1,
                                    name="{}_stage_{}".format(name, self.max_depth - d - 1),
                                    decision_type=self.decision_type,
                                    kernel=kernel,
                                    kernel_shape=self.kernel_shape,
                                    down_scale=self.max_depth - d - 1,
                                    feature_aggregation=self.feature_aggregation,
                                    clf=self.clf,
                                    clf_options=self.clf_options,
                                    data_reduction=self.data_reduction)
        return x1


class PyramidBoosting3D:
    def __init__(self,
                 n_estimators=3,
                 max_depth=5,
                 max_kernel_sum=5,
                 features_to_use="gray-color",
                 decision_type="kernel",
                 kernel_shape="square",
                 feature_aggregation="hist32",
                 norm_input=None,
                 clf="b_rf",
                 clf_options=None,
                 data_reduction=6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_kernel_sum = max_kernel_sum
        self.features_to_use = features_to_use
        self.norm_input = norm_input
        self.kernel_shape = kernel_shape
        self.decision_type = decision_type
        self.feature_aggregation = feature_aggregation

        self.clf = clf
        self.clf_options = clf_options
        self.data_reduction = data_reduction

    def build(self, width=None, height=None, initial_down_scale=None):
        xx = Input3DLayer(name="in",
                          features_to_use=self.features_to_use,
                          initial_down_scale=initial_down_scale,
                          width=width,
                          height=height)

        if self.norm_input is not None:
            pass

        for i in range(self.n_estimators):
            xx = self._build_single_unit(xx, name="estimator_{}".format(i))
        return xx

    def _build_single_unit(self, input_layer, name):
        x1 = input_layer

        for d in range(self.max_depth):
            kernel = (self.max_kernel_sum - 2 * d,
                      self.max_kernel_sum - 2 * d,
                      self.max_kernel_sum - 2 * d)
            x1 = get_decision_layer_3d(INPUTS=x1,
                                       decision_type=self.decision_type,
                                       name="{}_stage_{}".format(name, self.max_depth - d - 1),
                                       kernel=kernel,
                                       kernel_shape="ellipse",
                                       down_scale=self.max_depth - d - 1,
                                       feature_aggregation=self.feature_aggregation,
                                       clf=self.clf,
                                       clf_options=self.clf_options,
                                       data_reduction=self.data_reduction)
            x1 = BottleNeck3DLayer(INPUTS=x1, name="{}_bottle_neck_{}".format(name, self.max_depth - d - 1))
        return x1

