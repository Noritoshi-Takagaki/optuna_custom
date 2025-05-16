import optuna
from optuna.samplers._tpe.sampler import TPESampler
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions, _BatchedCategoricalDistributions
import numpy as np
import matplotlib.pyplot as plt

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

configurations = {"default","custom"}
#結果の格納
config_results = {}

roops = 30
all_runs = []

for seed_n in range(roops):
    try_sampler = TPESampler(seed = seed_n,multivariate = False)
    #try_sampler.beta = 5
    # スタディの作成
    study = optuna.create_study(sampler = try_sampler, direction = "minimize")
    best_value = float("inf")
    trial_values = []
    trials_n = 100
    # 最適化の実行
    study.optimize(objective, n_trials=trials_n)
    for _ in range(trials_n):
        trial = study.ask()
        value = objective(trial)
        study.tell(trial, value)
        best_value = min(best_value, value)
        trial_values.append(best_value)
    all_runs.append(trial_values)

config_results["default"] = np.array(all_runs)

for seed_n in range(roops):
    try_sampler = TPESampler(seed = seed_n,multivariate = False)
    try_sampler._below_distributions = {"x": [batched_distributions_below_distributions_for_x], 
                                        "c": [batched_distributions_below_distributions_for_c]}
    try_sampler._above_distributions = {"x": [batched_distributions_above_distributions_for_x], 
                                        "c": [batched_distributions_above_distributions_for_c]}
    #try_sampler.beta = 5
    # スタディの作成
    study = optuna.create_study(sampler = try_sampler, direction = "minimize")
    best_value = float("inf")
    trial_values = []
    trials_n = 100
    # 最適化の実行
    study.optimize(objective, n_trials=trials_n)
    for _ in range(trials_n):
        trial = study.ask()
        value = objective(trial)
        study.tell(trial, value)
        best_value = min(best_value, value)
        trial_values.append(best_value)
    all_runs.append(trial_values)

config_results["custom"] = np.array(all_runs)

# 結果の表示
for config_name, data in config_results.items():
    mean_curve = np.mean(data, axis=0)
    std_curve = np.std(data, axis=0)
    plt.plot(mean_curve, label=f"{config_name} (mean)")
    plt.fill_between(range(trials_n), mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

plt.xlabel("Trial")
plt.ylabel("Best Value So Far")
plt.title("Convergence Curves for Different Settings (Mean ± Std)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()