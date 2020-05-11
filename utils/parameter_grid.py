from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer


def f1_measure(pos_label=1, func_only=False):
    def f1_measure_func(y_true, y_pred):
        return f1_score(y_true, y_pred, pos_label=pos_label)
    if func_only:
        return f1_measure_func
    else:
        return make_scorer(f1_measure_func, greater_is_better=True)


def discard_score(pos_label=0, neg_label=1):
    def discard_score_func(y_true, y_pred):
        p0 = recall_score(y_true, y_pred, pos_label=pos_label)
        p1 = precision_score(y_true, y_pred, pos_label=neg_label)
        return 2 * (p0 * p1) / (p0 + p1 + 1e-7)
    return make_scorer(discard_score_func, greater_is_better=True)


def random_forrest_grid():
    return {'n_estimators': [10, 100, 200, 1000, 2500, 5000],
            'max_features': ['auto'],
            'max_depth': [None, 1, 2, 5, 10, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]}


def random_forrest_grid_estimators():
    return {'n_estimators': [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]}


def random_forrest_grid_fixed_estimators():
    return {'max_features': ['auto'],
            'max_depth': [None, 1, 2, 5, 10, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]}


def support_vector_machine_grid():
    return {"C": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "gamma": ["scale", "auto"],
            "shrinking": [True, False],
            "class_weight": [None, "balanced"]}


def decision_tree_grid():
    return {"criterion": ["gini"],
            "max_depth": [None, 1, 2, 5, 10, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'auto', 'sqrt']}


def xgboost_grid():
    return {'max_depth': [1, 2, 5, 10, 25, 50, 100],
            "booster": ["gbtree", "gblinear", "dart"],
            "tree_methode": "hist",
            "num_parallel_tree": [1, 2, 5, 10, 25, 50, 100],
            'min_child_weight': [1, 2, 3, 4],
            'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1.0],
            'n_estimators': [10, 100, 200, 1000, 2500, 5000],
            'subsample': [0.8, 0.9, 1],
            'max_delta_step': [0, 1, 2, 4]}


def xgboost_grid_small():
    return {"booster": ["gbtree"],
            "tree_methode": ["hist"],
            "num_parallel_tree": [5, 10, 15],
            'n_estimators': [5, 10, 25, 50, 75]}


def boosting_grid():
    return {"n_estimators": [2, 5, 10, 100, 200, 1000, 2500, 5000],
            'learning_rate': [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 1]}


def bagging_grid():
    return {"n_estimators": [2, 5, 10, 100, 200, 1000, 2500, 5000],
            "max_samples": [0.25, 0.5, 0.75, 1.0],
            "max_features": [0.25, 0.5, 0.75, 1.0],
            "bootstrap": [True, False],
            "bootstrap_features": [True, False]}


def sgd_grid():
    return {"loss": ["hinge", "log", "modified_huber", "squared_hinge",
                     "perceptron", "squared_loss", "huber",
                     "epsilon_insensitive", "squared_epsilon_insensitive"],
            "penalty": ["l2", "l1", "elasticnet"]}


def mlp_grid():
    return {"hidden_layer_sizes": [(30,), (50,), (100,), (100, 50), (100, 50, 30)],
            "activation": ["logistic", "tanh", "relu"],
            "solver": ["adam"],
            "learning_rate": ["adaptive"]}


def gradient_boosting_grid():
    return {'learning_rate': [0.01, 0.02, 0.03],
            'subsample': [0.9, 0.5, 0.2],
            'n_estimators': [100, 500, 1000],
            'max_depth': [4, 6, 8]}

