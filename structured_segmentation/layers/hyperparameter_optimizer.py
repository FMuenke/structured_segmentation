from sklearn.model_selection import ParameterGrid


class HyperParameterOptimizer:
    def __init__(self, LayerPrototype, parameter_grid):
        self.layer_prototype = LayerPrototype
        self.pg = parameter_grid

        for p in self.pg:
            if type(self.pg[p]) is not list:
                self.pg[p] = [self.pg[p]]

        self.selected_layer = None
        self.index = 0

    def set_index(self, i):
        self.index = i

    def fit(self, train_tags, validation_tags):
        max_score = 0
        if self.selected_layer is None:
            print("Start HyperParameter Optimization:")
            for kwargs in list(ParameterGrid(self.pg)):
                try:
                    layer = self.layer_prototype(**kwargs)
                    score = layer.fit(train_tags, validation_tags)

                    if max_score < score:
                        max_score = score
                        self.selected_layer = layer
                        self.selected_layer.set_index(self.index)
                except Exception as e:
                    print(e)
                    pass
            print("HyperParameter Optimization complete!")
            print("Selected Model: {}".format(self.selected_layer))

    def save(self, model_path):
        self.selected_layer.save(model_path)

    def inference(self, x_input, interpolation="nearest"):
        return self.selected_layer.inference(x_input, interpolation)
