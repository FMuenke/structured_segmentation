import os
import numpy as np
from tqdm import tqdm

from skimage.measure import label, regionprops

from structured_segmentation.learner.internal_classifier import InternalClassifier
from structured_segmentation.layers.layer_operations import resize
from structured_segmentation.utils.utils import check_n_make_dir, save_dict


class ObjectSelectionLayer:
    layer_type = "OBJECT_SELECTION_LAYER"

    def __init__(self,
                 INPUTS,
                 name,
                 clf="rf",
                 clf_options=None,
                 data_reduction=0):

        self.name = str(name)
        if type(INPUTS) is not list:
            INPUTS = [INPUTS]
        self.previous = INPUTS

        self.index = 0

        for i, p in enumerate(self.previous):
            p.set_index(i)
        self.max_num_samples = 500000

        self.opt = {
            "name": self.name,
            "layer_type": self.layer_type,
        }

        if clf_options is None:
            clf_options = {"type": clf}
        else:
            clf_options["type"] = clf
        self.clf = InternalClassifier(opt=clf_options)
        self.clf.new()
        self.data_reduction = data_reduction

    def __str__(self):
        return "{} - {} - {}".format(self.layer_type, self.name, self.clf)

    def inference(self, x_input, interpolation="nearest"):
        x_img, x_pass = self.get_features(x_input)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
        x_img = np.reshape(x_img, (x_height * x_width, -1))
        probs = self.clf.predict_proba(x_img)
        n_classes = probs.shape[1]
        y_img = []
        for i in range(n_classes):
            y_i_img = np.reshape(probs[:, i], (x_height, x_width, 1))
            y_img.append(y_i_img)
        y_img = np.concatenate(y_img, axis=2)

        x_img_pass = resize(x_pass, width=o_width, height=o_height, interpolation="nearest")
        y_img = resize(y_img, width=o_width, height=o_height, interpolation=interpolation)

        if len(x_img_pass.shape) < 3:
            x_img_pass = np.expand_dims(x_img_pass, axis=2)
        if len(y_img.shape) < 3:
            y_img = np.expand_dims(y_img, axis=2)
        y_img = np.concatenate([x_img_pass, y_img], axis=2)
        return y_img

    def predict(self, x_input):
        x_img, x_pass = self.get_features(x_input)
        o_height, o_width = x_pass.shape[:2]
        x_height, x_width = x_img.shape[:2]
        x_img = x_img[:, :, -1]
        label_img = label(x_img)
        props = regionprops(label_img)
        x_props = []
        y_img = np.zeros((x_height, x_width))
        for p in props:
            x_p = self.properties_to_features(p)
            x_props.append(x_p)
        if len(x_props) == 0:
            return y_img
        x_props = np.concatenate(x_props, axis=0)
        y_props = self.clf.predict(x_props)
        for i, p in enumerate(props):
            y_img[label_img == p["label"]] = y_props[i]
        y_img = resize(y_img, width=o_width, height=o_height, interpolation="nearest")
        return y_img

    def get_features(self, x_input):
        x = []
        for p in self.previous:
            x_p = p.inference(x_input, interpolation="linear")
            if len(x_p.shape) < 3:
                x_p = np.expand_dims(x_p, axis=2)
            x.append(x_p)
        x = np.concatenate(x, axis=2)
        x_pass = np.copy(x)
        x = x.astype(np.float32)
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        return x, x_pass

    def properties_to_features(self, p):
        x_p = [
            p["area"],
            p["convex_area"],
            p["solidity"],
            p["centroid"][0],
            p["centroid"][1],
            p["eccentricity"],
            p["euler_number"],
        ]
        return np.reshape(x_p, (1, -1))

    def get_x_y(self, tag_set, reduction_factor=0):
        x = None
        y = None
        for t in tqdm(tag_set):
            use_sample = True
            if reduction_factor > 1:
                if not np.random.randint(0, reduction_factor):
                    use_sample = False

            if use_sample:
                x_img = t.load_x()
                x_img, _ = self.get_features(x_img)
                x_img = x_img[:, :, -1]
                h_img, w_img = x_img.shape[:2]
                y_img = t.load_y([h_img, w_img])

                label_img = label(x_img)
                props = regionprops(label_img)
                for p in props:
                    x_p = self.properties_to_features(p)
                    counts = np.bincount(y_img[label_img == p["label"]].astype(np.int))
                    y_p = np.array(np.argmax(counts))
                    y_p = np.reshape(y_p, (1, ))

                    if x is None:
                        x = x_p
                        y = y_p
                    else:
                        x = np.append(x, x_p, axis=0)
                        y = np.append(y, y_p, axis=0)

        x = x.astype(np.float32)
        return x, y

    def fit(self, train_tags, validation_tags):
        for p in self.previous:
            p.fit(train_tags, validation_tags)

        print("Collecting Features for Stage: {}".format(self))
        print("Data is reduced by factor: {}".format(self.data_reduction))
        x_train, y_train = self.get_x_y(train_tags, reduction_factor=self.data_reduction)
        if len(np.unique(y_train)) == 1:
            print("Could Not Train a Useful ObjectSelector")
            return None
        x_val, y_val = self.get_x_y(validation_tags)

        n_samples_train, n_features = x_train.shape
        n_samples_val = x_val.shape[0]
        print("DataSet has {} Samples (Train: {} / Validation: {}) with {} features.".format(
            n_samples_train + n_samples_val, n_samples_train, n_samples_val, n_features
        ))
        self.clf.fit(x_train, y_train)
        return self.clf.evaluate(x_val, y_val)

    def save(self, model_path):
        model_path = os.path.join(model_path, self.layer_type + "-" + self.name)
        check_n_make_dir(model_path)
        self.clf.save(model_path)
        self.opt["index"] = self.index
        save_dict(self.opt, os.path.join(model_path, "opt.json"))
        for p in self.previous:
            p.save(model_path)

    def load(self, model_path):
        self.clf.load(model_path)

    def set_index(self, i):
        self.index = i
