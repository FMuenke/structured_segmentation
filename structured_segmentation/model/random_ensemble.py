import numpy as np

from structured_segmentation.layers import StructuredClassifierLayer
from structured_segmentation.layers import InputLayer
from structured_segmentation.layers import VotingLayer
from structured_segmentation.layers import NormalizationLayer
from structured_segmentation.layers import BottleNeckLayer

from structured_segmentation.model.model import Model
from structured_segmentation.model.graph import Graph


def randomize_options(max_val):
    k_y = max(np.random.randint(max_val), 1)
    k_x = int(max_val - k_y)
    return k_y, k_x


class RandomEnsemble(Model):
    def __init__(self,
                 image_width=None,
                 image_height=None,
                 initial_image_down_scale=None,
                 n_estimators=5,
                 max_kernel_sum=10,
                 max_stride_sum=1,
                 max_down_scale=6,
                 features_to_use="gray-color",
                 norm_input=None,
                 clf="hgb",
                 kernel_shape="ellipse",
                 clf_options=None,
                 data_reduction=3):
        self.n_estimators = n_estimators
        self.max_kernel_sum = max_kernel_sum
        self.max_stride_sum = max_stride_sum
        self.max_down_scale = max_down_scale
        self.features_to_use = features_to_use
        self.norm_input = norm_input
        self.kernel_shape = kernel_shape
        self.data_reduction = data_reduction

        self.clf = clf
        self.clf_options = clf_options
        super(RandomEnsemble, self).__init__()

        self.model = self.build(
            image_width,
            image_height,
            initial_image_down_scale,
        )

    def build(self, width=None, height=None, initial_down_scale=None):
        tree = tree = InputLayer(
            name="input_tree",
            features_to_use=self.features_to_use,
            width=width, height=height,
            initial_down_scale=initial_down_scale
        )
        if self.norm_input is not None:
            tree = NormalizationLayer(
                INPUTS=tree,
                name="norm_tree",
                norm_option=self.norm_input
            )
        for i in range(self.n_estimators):
            tree = StructuredClassifierLayer(
                INPUTS=tree,
                name="tree_{}".format(i),
                kernel=randomize_options(self.max_kernel_sum),
                strides=randomize_options(self.max_stride_sum),
                kernel_shape=self.kernel_shape,
                down_scale=np.random.randint(self.max_down_scale),
                data_reduction=self.data_reduction,
                clf=self.clf,
                clf_options=self.clf_options
            )

        tree = StructuredClassifierLayer(
            INPUTS=tree,
            name="final_layer",
            kernel=(3, 3),
            clf=self.clf,
            clf_options=self.clf_options
        )
        return Graph(layer_stack=tree)
