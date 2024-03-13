from structured_segmentation.layers import StructuredClassifierLayer
from structured_segmentation.layers import StructuredEncoderLayer
from structured_segmentation.layers import InputLayer
from structured_segmentation.layers import VotingLayer
from structured_segmentation.layers import NormalizationLayer
from structured_segmentation.layers import BottleNeckLayer

from structured_segmentation.model.model import Model

from structured_segmentation.model.graph import Graph


class Ensemble(Model):
    def __init__(self,
                 image_width=None,
                 image_height=None,
                 initial_image_down_scale=None,
                 output_aggregation_options="boosting",
                 kernel=5,
                 strides=2,
                 max_down_scale=3,
                 features_to_use="gray-color",
                 tree_type="kernel",
                 norm_input=None,
                 clf="extra_tree",
                 kernel_shape="ellipse",
                 data_reduction=0.20):
        self.kernel = kernel
        self.strides = strides
        self.max_down_scale = max_down_scale
        self.features_to_use = features_to_use
        self.norm_input = norm_input
        self.kernel_shape = kernel_shape
        self.data_reduction = data_reduction
        self.tree_type = tree_type

        self.clf = clf
        super(Ensemble, self).__init__()

        self.model = self.build(
            image_width,
            image_height,
            initial_image_down_scale,
            output_aggregation_options
        )

    def build(self, width=None, height=None, initial_down_scale=None, output_option="voting"):

        trees = []    
        for scale in range(0, self.max_down_scale):
            tree = InputLayer(
                name=f"INPUT_{scale}",
                features_to_use=self.features_to_use,
                width=width, height=height,
                initial_down_scale=initial_down_scale
            )
            if self.norm_input is not None:
                tree = NormalizationLayer(
                    INPUTS=tree,
                    name=f"norm_tree_{scale}",
                    norm_option=self.norm_input
                )
            tree = StructuredEncoderLayer(
                INPUTS=tree,
                name=f"tree_{scale}",
                kernel=(self.kernel, self.kernel),
                strides=(self.strides, self.strides),
                kernel_shape=self.kernel_shape,
                down_scale=scale,
                enc=self.clf,
                data_reduction=self.data_reduction
            )
            trees.append(tree)


        if output_option == "voting":
            trees = VotingLayer(INPUTS=trees, name="voting")
        elif output_option == "boosting":
            bottle_neck = BottleNeckLayer(INPUTS=trees, name="cls_preparation")
            trees = StructuredClassifierLayer(
                INPUTS=bottle_neck,
                kernel=(1, 1),
                name="boosting",
                clf="hgb",
                data_reduction=self.data_reduction,
            )
        else:
            raise Exception("Unknown Option : {}".format(output_option))
        return Graph(layer_stack=trees)
