from structured_classifier.decision_layer import DecisionLayer


def down_pyramid(input_layer, name, kernel=(5, 5), clf="b_rf", depth=5, repeat=1, clf_options=None):
    x1 = input_layer

    for d in range(depth):
        for r in range(repeat):
            x1 = DecisionLayer(INPUTS=x1,
                               name="{}_down_{}_{}".format(name, d, r),
                               kernel=kernel,
                               kernel_shape="ellipse",
                               down_scale=d,
                               clf=clf,
                               clf_options=clf_options,
                               data_reduction=3)
    return x1


def up_pyramid(input_layer, name, kernel=(5, 5), clf="b_rf", depth=5, repeat=1, clf_options=None):
    x1 = input_layer

    for d in range(depth):
        for r in range(repeat):
            x1 = DecisionLayer(INPUTS=x1,
                               name="{}_up_{}_{}".format(name, depth - d - 1, r),
                               kernel=kernel,
                               kernel_shape="ellipse",
                               down_scale=depth - d - 1,
                               clf=clf,
                               clf_options=clf_options,
                               data_reduction=3)
    return x1


def u_layer(input_layer: object, name: object, kernel: object = (5, 5), clf: object = "b_rf", depth: object = 5, repeat: object = 1, clf_options: object = None) -> object:
    x1 = input_layer

    for d in range(depth):
        for r in range(repeat):
            x1 = DecisionLayer(INPUTS=x1,
                               name="{}_down_{}_{}".format(name, d, r),
                               kernel=kernel,
                               kernel_shape="ellipse",
                               down_scale=d,
                               clf=clf,
                               clf_options=clf_options,
                               data_reduction=3)

    for r in range(repeat):
        x1 = DecisionLayer(INPUTS=x1,
                           name="{}_latent_{}".format(name, r),
                           kernel=kernel,
                           kernel_shape="ellipse",
                           down_scale=depth,
                           clf=clf,
                           clf_options=clf_options,
                           data_reduction=3)

    for d in range(depth):
        for r in range(repeat):
            x1 = DecisionLayer(INPUTS=x1,
                               name="{}_up_{}_{}".format(name, depth - d - 1, r),
                               kernel=kernel,
                               kernel_shape="ellipse",
                               down_scale=depth - d - 1,
                               clf=clf,
                               clf_options=clf_options,
                               data_reduction=3)

    return x1
