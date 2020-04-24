import argparse

from data_structure.data_set import DataSet
from structured_classifier.model import Model

from structured_classifier.decision_layer import DecisionLayer
from structured_classifier.input_layer import InputLayer
from structured_classifier.global_context_layer import GlobalContextLayer


def main(args_):
    color_coding = {
        # "man_hole": [[1, 1, 1], [0, 255, 0]],
        # "crack": [[3, 3, 3], [255, 255, 0]],
        "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        # "heart": [[4, 4, 4], [0, 255, 0]],
        # "muscle": [[255, 255, 255], [255, 0, 0]],
        # "shadow": [[1, 1, 1], [255, 0, 0]],
        # "filled_crack": [[2, 2, 2], [0, 255, 0]],
    }

    randomized_split = False
    train_test_ratio = 0.25

    df = args_.dataset_folder
    mf = args_.model_folder

    input_1 = InputLayer("input_1", "gray-color")
    x1 = DecisionLayer(INPUTS=input_1, name="decision_1", kernel=(5, 5), down_scale=1)
    x1 = GlobalContextLayer(INPUTS=x1, name="glob_context", down_scale=2)
    # x1 = DecisionLayer(INPUTS=x1, name="decision_2", kernel=(5, 5), down_scale=2)
    # x1 = DecisionLayer(INPUTS=x1, name="decision_3", kernel=(5, 5), down_scale=3)
    # x1 = DecisionLayer(INPUTS=x1, name="decision_4", kernel=(5, 5), down_scale=4)
    # x1 = DecisionLayer(INPUTS=x1, name="decision_5", kernel=(5, 5), down_scale=3)

    f1 = DecisionLayer(INPUTS=x1, name="final_decision", kernel=(1, 1), down_scale=2)

    model = Model(graph=f1)

    d_set = DataSet(df, color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set, percentage=train_test_ratio, random=randomized_split)

    model.fit(train_set, validation_set)
    model.save(mf)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with predictions",
    )
    parser.add_argument(
        "--model_folder", "-m", default="./test", help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
