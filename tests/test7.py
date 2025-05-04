import optuna
from optuna.samplers._tpe.sampler import TPESampler
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions, _BatchedCategoricalDistributions
import numpy as np

def objective(trial):
    x = trial.suggest_float("x", -4.5, 4.5)
    c = trial.suggest_categorical("c", ("False", "True"))

    if c == "True":
        # 通常の Beale 関数（最適解が x=3 にあるように設計）
        cf=0
    else:
        # c=False のときは最小値がずれるように変形
        cf=1
    return x ** 2 - 6 * x + 9 + 10 * cf

batched_distributions_below_distributions_for_x = _BatchedTruncNormDistributions(
    mu=np.array([3.0]),     # 平均値の配列
    sigma=np.array([0.3]),  # 標準偏差の配列
    low=-4.5,
    high=4.5
)

batched_distributions_above_distributions_for_x = _BatchedTruncNormDistributions(
    mu=np.array([-2.0]),    # 平均値の配列
    sigma=np.array([2.5]),  # 標準偏差の配列
    low=-4.5,
    high=4.5
)

batched_distributions_below_distributions_for_c = _BatchedCategoricalDistributions(
    weights=np.array([[0.05, 0.95]])
)

batched_distributions_above_distributions_for_c = _BatchedCategoricalDistributions(
    weights=np.array([[0.95, 0.05]])
)

try_sampler = TPESampler()

try_sampler._below_distributions = {"x": [batched_distributions_below_distributions_for_x], 
                                    "c": [batched_distributions_below_distributions_for_c]}
try_sampler._above_distributions = {"x": [batched_distributions_above_distributions_for_x], 
                                    "c": [batched_distributions_above_distributions_for_c]}

# スタディの作成
study = optuna.create_study(sampler = try_sampler, direction = "minimize")

# 最適化の実行
study.optimize(objective, n_trials=100)

# 結果の表示
print(f"Best objective value: {study.best_value}")
print(f"Best parameter: {study.best_params}")
optuna.visualization.plot_optimization_history(study).show()