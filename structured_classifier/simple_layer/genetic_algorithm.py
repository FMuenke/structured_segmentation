import numpy as np
import random
from multiprocessing.pool import Pool

from sklearn.model_selection import ParameterGrid

from structured_classifier.simple_layer.pipeline import Pipeline
from structured_classifier.simple_layer.image_processing_operations import LIST_OF_OPERATIONS


def eval_pipeline(args):
    evaluated_pipelines = []
    for bundle in args:
        pl, x_img, y_img = bundle
        pl.eval(x_img, y_img)
        evaluated_pipelines.append(pl)
    return evaluated_pipelines


def flatten_list(t):
    flat_list = [item for sublist in t for item in sublist]
    return flat_list


class MutationChamber:
    def __init__(self, individuals, number_to_generate):
        self.individuals = individuals
        self.number_to_generate = number_to_generate
        self.to_do = number_to_generate

    def __iter__(self):
        self.to_do = self.number_to_generate
        return self

    def __len__(self):
        return self.number_to_generate

    def __next__(self):
        if self.to_do <= 0:
            raise StopIteration
        self.to_do -= 1

        i1, i2 = np.random.choice(self.individuals, size=2)
        config = []
        for list_of_op in zip(i1.config, i2.config):
            sel = np.random.randint(len(list_of_op))
            config.append(list_of_op[sel])
        return Pipeline(config, selected_layer=np.random.choice([i1.selected_layer, i2.selected_layer]))


class Population:
    def __init__(self, number_of_individuals, individual_pool, use_multi_processing):
        self.number_of_individuals = number_of_individuals
        self.individual_pool = individual_pool
        self.use_multi_processing = use_multi_processing
        self.multiprocessing_chunk_size = 100

        self.keep_percentage = 0.3
        self.new_percentage = 0.2
        self.mut_percentage = 0.5

        self.individuals = self.spawn_individuals(self.number_of_individuals)

    def spawn_individuals(self, n):
        return [self.individual_pool.pop(random.randrange(len(self.individual_pool))) for _ in range(n)]

    def score_individuals(self, x_img, y_img):
        if self.use_multi_processing:
            tasks = [[pl, x_img, y_img] for pl in self.individuals]
            n = self.multiprocessing_chunk_size
            tasks_bundled = [tasks[i:i + n] for i in range(0, len(tasks), n)]
            with Pool() as p:
                bundled_pipelines = p.map(eval_pipeline, tasks_bundled)
            self.individuals = flatten_list(bundled_pipelines)
        else:
            for pl in self.individuals:
                pl.eval(x_img, y_img)

    def evolve(self):
        top_k = int(self.number_of_individuals * self.keep_percentage)
        new_k = int(self.number_of_individuals * self.new_percentage)
        mut_k = int(self.number_of_individuals * self.mut_percentage)

        scored_individuals = [(pl, pl.summarize()) for pl in self.individuals]
        top_individuals_and_scores = sorted(scored_individuals, key=lambda x: x[1], reverse=True)[:top_k]
        # print(top_individuals_and_scores[0])
        top_individuals = [pl for pl, score in top_individuals_and_scores]
        new_individuals = self.spawn_individuals(new_k)

        mutation_chamber = MutationChamber(individuals=top_individuals, number_to_generate=mut_k)
        mut_individuals = [pl for pl in mutation_chamber]

        self.individuals = top_individuals + new_individuals + mut_individuals
        return top_individuals_and_scores[0]


class GeneticAlgorithmOptimizer:
    def __init__(self, operations, selected_layer, use_multi_processing):
        self.operations = operations
        self.selected_layer = selected_layer
        self.use_multi_processing = use_multi_processing

        self.population_size = 1000
        self.iter_per_image = 4

        if type(self.selected_layer) is not list:
            self.pipelines = [Pipeline(config, self.selected_layer) for config in self.build_configs()]
        else:
            self.pipelines = []
            for selected_index in self.selected_layer:
                self.pipelines += [Pipeline(config, selected_index) for config in self.build_configs()]
        print("Evaluating - {} - Configurations".format(len(self.pipelines)))

        self.population = Population(
            self.population_size,
            self.pipelines,
            use_multi_processing=self.use_multi_processing
        )
        self.best_pipeline = None
        self.best_score = None

    def build_configs(self):
        list_of_configs = []
        possible_configs = {Op.key: Op.list_of_parameters for Op in LIST_OF_OPERATIONS}
        selected_configs = {op: possible_configs[op] for op in possible_configs if op in self.operations}
        for parameters in list(ParameterGrid(selected_configs)):
            cfg = [[op, parameters[op]] for op in self.operations]
            list_of_configs.append(cfg)
        return list_of_configs

    def step(self, x_img, y_img):
        for i in range(self.iter_per_image):
            self.population.score_individuals(x_img, y_img)
            self.best_pipeline, self.best_score = self.population.evolve()

    def summarize(self):
        return self.best_pipeline, self.best_score
