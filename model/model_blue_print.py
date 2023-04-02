from utils.utils import check_n_make_dir


class ModelBluePrint:
    def __init__(self):
        self.model = None

    def fit(self, train_set, validation_set=None):
        self.model.fit(train_set, validation_set)

    def save(self, path):
        check_n_make_dir(path)
        self.model.save(path)

    def load(self, path):
        self.model.load(path)
