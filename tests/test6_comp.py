import optuna
from optuna.samplers._tpe.sampler import TPESampler
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions
import numpy as np

def objective(trial):
    x = trial.suggest_float("x", -4.5, 4.5)
    y = trial.suggest_float("y", -4.5, 4.5)

    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2
#"""
batched_distributions_below_distributions_for_x = _BatchedTruncNormDistributions(
    mu=np.array([3.0]),   # 平均値の配列
    sigma=np.array([0.01]), # 標準偏差の配列
    low=-4.5,
    high=4.5
)

batched_distributions_above_distributions_for_x = _BatchedTruncNormDistributions(
    mu=np.array([-2.0]),   # 平均値の配列
    sigma=np.array([2.5]), # 標準偏差の配列
    low=-4.5,
    high=4.5
)

batched_distributions_below_distributions_for_y = _BatchedTruncNormDistributions(
    mu=np.array([0.5]),   # 平均値の配列
    sigma=np.array([0.01]), # 標準偏差の配列
    low=-4.5,
    high=4.5
)

batched_distributions_above_distributions_for_y = _BatchedTruncNormDistributions(
    mu=np.array([4.0]),   # 平均値の配列
    sigma=np.array([2.5]), # 標準偏差の配列
    low=-4.5,
    high=4.5
)

try_sampler = TPESampler(n_startup_trials = 0, seed = 1,multivariate = True)

try_sampler._below_distributions = {"x": [batched_distributions_below_distributions_for_x], 
                                    "y": [batched_distributions_below_distributions_for_y]}
try_sampler._above_distributions = {"x": [batched_distributions_above_distributions_for_x], 
                                    "y": [batched_distributions_above_distributions_for_y]}

# スタディの作成
study = optuna.create_study(sampler = try_sampler, direction = "minimize")

# 最適化の実行
study.optimize(objective, n_trials=100)

# 結果の表示
print(f"Best objective value: {study.best_value}")
print(f"Best parameter: {study.best_params}")
#optuna.visualization.plot_optimization_history(study).show()