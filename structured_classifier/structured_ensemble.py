import os
import numpy as np
from tqdm import tqdm

from structured_classifier.structured_classifier import StructuredClassifier


class StructuredEnsemble:
    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.member = None

    def save(self):
        for m in self.member:
            m.save()

    def load(self):
        self.member = {}
        for member_id in os.listdir(self.model_folder):
            m_path = os.path.join(self.model_folder, member_id)
            if os.path.isdir(m_path):
                s_cls = StructuredClassifier(m_path)
                s_cls.load()
                self.member[member_id] = s_cls

    def get_x_y(self, tag_set):
        x = []
        y = []
        for member_id in self.member:
            print("Feature Extraction for {}...".format(member_id))
            m = self.member[member_id]
            for t in tqdm(tag_set):
                x_img = t.load_x()
                h_img, w_img = x_img.shape[:2]
                x_img = m.inference(x_img, interpolation="linear")
                y_img = t.load_y([h_img, w_img])

                x_img = np.reshape(x_img, (h_img * w_img, -1))
                y_img = np.reshape(y_img, h_img * w_img)

                x.append(x_img)
                y.append(y_img)
            x = np.concatenate(x, axis=0)
            y = np.concatenate(y, axis=0)
            return x, y

    def _fit_master(self, train_set, validation_set, ensemble_cfg):
        master = StructuredClassifier(self.model_folder)
        master.build(ensemble_cfg.master_cfg)

    def fit(self, train_set, validation_set, ensemble_cfg):
        self.member = []
        for cfg in ensemble_cfg.built_configs:
            mf = os.path.join(self.model_folder, cfg.opt["member_id"])
            m = StructuredClassifier(mf)
            m.fit(train_set, validation_set, data_reduction_factor=200)
            self.member[cfg.opt["member_id"]] = m
