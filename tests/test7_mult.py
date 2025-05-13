import optuna
from optuna.samplers._tpe.sampler import TPESampler
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions, _BatchedCategoricalDistributions
import numpy as np

def objective(trial):
    x0 = trial.suggest_categorical("x0", [0, 1])
    x1 = trial.suggest_categorical("x1", [0, 1])
    x2 = trial.suggest_categorical("x2", [0, 1])
    x3 = trial.suggest_categorical("x3", [0, 1])
    x4 = trial.suggest_categorical("x4", [0, 1])
    x5 = trial.suggest_categorical("x5", [0, 1])
    x6 = trial.suggest_categorical("x6", [0, 1])
    x7 = trial.suggest_categorical("x7", [0, 1])

    score = 0
    if x0 == 1 and x1 == 1:
        score += 5
    if x2 == 1 and x3 == 0:
        score += 3
    if x4 == x5:
        score += 2
    if (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7) >= 6:
        score += 4
    if x6 ^ x7:
        score += 1.5
    if [x0, x1, x2, x3] == [1, 1, 1, 1]:
        score -= 10

    return -score  # 最小化


pprb_0 = _BatchedCategoricalDistributions(
    weights=np.array([[0.1, 0.9]])
)
pprb_1 = _BatchedCategoricalDistributions(
    weights=np.array([[0.1, 0.9]])
)
pprb_2 = _BatchedCategoricalDistributions(
    weights=np.array([[0.1, 0.9]])
)
pprb_3 = _BatchedCategoricalDistributions(
    weights=np.array([[0.9, 0.1]])
)
pprb_4 = _BatchedCategoricalDistributions(
    weights=np.array([[0.9, 0.1]])
)
pprb_5 = _BatchedCategoricalDistributions(
    weights=np.array([[0.9, 0.1]])
)
pprb_6 = _BatchedCategoricalDistributions(
    weights=np.array([[0.9, 0.1]])
)
pprb_7 = _BatchedCategoricalDistributions(
    weights=np.array([[0.1, 0.9]])
)

ppra_0 = _BatchedCategoricalDistributions(
    weights=np.array([[0.9, 0.1]])
)
ppra_1 = _BatchedCategoricalDistributions(
    weights=np.array([[0.9, 0.1]])
)
ppra_2 = _BatchedCategoricalDistributions(
    weights=np.array([[0.9, 0.1]])
)
ppra_3 = _BatchedCategoricalDistributions(
    weights=np.array([[0.1, 0.9]])
)
ppra_4 = _BatchedCategoricalDistributions(
    weights=np.array([[0.1, 0.9]])
)
ppra_5 = _BatchedCategoricalDistributions(
    weights=np.array([[0.1, 0.9]])
)
ppra_6 = _BatchedCategoricalDistributions(
    weights=np.array([[0.1, 0.9]])
)
ppra_7 = _BatchedCategoricalDistributions(
    weights=np.array([[0.9, 0.1]])
)

try_sampler = TPESampler(seed = 1,multivariate = False)
"""
try_sampler._below_distributions = {"x0": [pprb_0], 
                                    "x1": [pprb_1],
                                    "x2": [pprb_2],
                                    "x3": [pprb_3],
                                    "x4": [pprb_4],
                                    "x5": [pprb_5],
                                    "x6": [pprb_6],
                                    "x7": [pprb_7]}
try_sampler._above_distributions = {"x0": [ppra_0], 
                                    "x1": [ppra_1],
                                    "x2": [ppra_2],
                                    "x3": [ppra_3],
                                    "x4": [ppra_4],
                                    "x5": [ppra_5],
                                    "x6": [ppra_6],
                                    "x7": [ppra_7]}
"""
#try_sampler.beta = 5
# スタディの作成
study = optuna.create_study(sampler = try_sampler, direction = "minimize")

# 最適化の実行
study.optimize(objective, n_trials=100)

# 結果の表示
print(f"Best objective value: {study.best_value}")
print(f"Best parameter: {study.best_params}")
optuna.visualization.plot_optimization_history(study).show()