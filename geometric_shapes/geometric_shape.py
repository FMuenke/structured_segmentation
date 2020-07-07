from geometric_shapes.ellipse import Ellipse
from geometric_shapes.ellipse_al import EllipseAl
from geometric_shapes.circle import Circle
from geometric_shapes.rectangle import Rectangle
from geometric_shapes.centroid import Centroid
from geometric_shapes.arbitrary import Arbitrary


def get_shape(target_shape):
    if target_shape == "ellipse":
        return Ellipse()
    if target_shape == "centroid":
        return Centroid()
    if target_shape == "circle":
        return Circle()
    if target_shape == "rectangle":
        return Rectangle()
    if target_shape == "ellipse_al":
        return EllipseAl()
    if target_shape == "arbitrary":
        return Arbitrary()
    raise ValueError("Shape - {} - not implemented yet".format(target_shape))


class GeometricShape:
    def __init__(self):
        # num_targets: Indicator for the code. Number of outputs of the neural network
        self.num_targets = None

    def get_label_map(self, parameters, height, width):
        # This function implements the conversion from a set of parameters to a label_map
        label_map = None
        # numpy array of size (height, width, 1)
        # Where the object is marked with 1 and background is marked 0
        return label_map

    def get_parameters(self, label_map):
        # This function implements the conversion from a label_map to a set of parameters
        parameters = [None, None, None]
        return parameters

    def eval(self, gtr_mao, pre_map):
        # This function generates a value to judge the performance of the Neural Network
        # If not implemented: just return 0
        return 0
