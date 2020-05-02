from structured_classifier.decision_layer import DecisionLayer


def u_layer(input_layer, name, kernel=(5, 5), clf="b_rf", depth=5, repeat=1, clf_options=None):
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
                               name="{}_up_{}_{}".format(name, depth - d, r),
                               kernel=kernel,
                               kernel_shape="ellipse",
                               down_scale=depth - d,
                               clf=clf,
                               clf_options=clf_options,
                               data_reduction=3)

    return x1
