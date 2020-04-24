import os
import json
from data_structure.folder import Folder

from utils import parameter_grid as pg


class Config:
    def __init__(self, opt, color_coding):
        self.color_coding = color_coding
        self.opt = opt
        self.data_reduction_factor = None
        self.train_test_ratio = None
        self.randomized_split = None


def build_cfg():
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

    opt = {
        "features_to_use": ["gray-lm"],
        "down_scale": 2,
        "look_up_window": 3,
        "classifier_opt": {
            "type": "rf",
            "n_estimators": 200,
            # "param_grid": pg.decision_tree_grid(),
        }
    }

    cfg = Config(opt, color_coding)
    cfg.data_reduction_factor = 2
    cfg.train_test_ratio = 0.25
    cfg.randomized_split = False
    return cfg


def load_config(model_dir):
    print("Load cfg from model directory")
    color_coding_path = os.path.join(model_dir, "color_coding.json")
    opt_path = os.path.join(model_dir, "opt.json")
    cfg = Config()
    cfg.color_coding = load_dict(color_coding_path)
    cfg.opt = load_dict(opt_path)
    return cfg


def save_config(model_dir, cfg):
    print("config.pickle is saved to {}".format(model_dir))
    fol = Folder(model_dir)
    fol.check_n_make_dir()
    color_coding_path = os.path.join(model_dir, "color_coding.json")
    opt_path = os.path.join(model_dir, "opt.json")
    save_dict(cfg.color_coding, color_coding_path)
    save_dict(cfg.opt, opt_path)


def save_dict(dict_to_save, path_to_save):
    with open(path_to_save, "w") as f:
        j_file = json.dumps(dict_to_save)
        f.write(j_file)


def load_dict(path_to_load):
    with open(path_to_load) as json_file:
        dict_to_load = json.load(json_file)
    return dict_to_load