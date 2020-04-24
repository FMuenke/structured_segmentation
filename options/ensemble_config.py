from options.config import Config


class EnsembleConfig:
    def __init__(self):
        self.color_coding = {
            "man_hole": [[1, 1, 1], [0, 255, 0]],
            # "crack": [[3, 3, 3], [255, 255, 0]],
            # "heart": [[4, 4, 4], [0, 255, 0]],
            # "muscle": [[255, 255, 255], [255, 0, 0]],
            # "heart": [[4, 4, 4], [0, 255, 0]],
            # "muscle": [[255, 255, 255], [255, 0, 0]],
            # "shadow": [[1, 1, 1], [255, 0, 0]],
            # "filled_crack": [[2, 2, 2], [0, 255, 0]],
        }

        self.classifier_opt = {
                    "type": "rf",
                    "n_estimators": 200,
                    # "param_grid": pg.decision_tree_grid(),
                }

        self.member_opt = {
            1: {
                "id_0": [["gray-color"], 5, None],
                # "id_1": [["gray-color"], 5, None],
            },
            3: {
                "id_0": [["gray-color"], 5, None],
                # "id_1": [["gray-color"], 5, None],
            }
        }
        self.built_configs = []

        self.data_reduction_factor = {
            0: 800,
            1: 200,
            2: 50,
            3: 10,
        }
        self.train_test_ratio = 0.4
        self.randomized_split = False

    def build(self):
        for stage in self.member_opt:
            for member_id in self.member_opt[stage]:
                opt = {
                    "member_id": member_id,
                    "features_to_use": self.member_opt[stage][member_id][0],
                    "down_scale": stage,
                    "look_up_window": self.member_opt[stage][member_id][1],
                    "look_up_window_gradient": self.member_opt[stage][member_id][2],
                    "classifier_opt": self.classifier_opt
                }
                cfg = Config(opt=opt, color_coding=self.color_coding)
                cfg.data_reduction_factor = self.data_reduction_factor[stage]
                self.built_configs.append(cfg)
