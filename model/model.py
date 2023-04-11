from utils.utils import check_n_make_dir


class Model:
    def __init__(self):
        self.model = None

    def fit(self, train_set, validation_set=None):
        self.model.fit(train_set, validation_set)

    def evaluate(self, test_tags, color_coding, results_folder, is_unsupervised=False):
        self.model.evaluate(test_tags, color_coding, results_folder, is_unsupervised)

    def predict(self, image):
        self.model.predict(image)

    def save(self, path):
        check_n_make_dir(path)
        self.model.save(path)

    def load(self, path):
        self.model.load(path)
