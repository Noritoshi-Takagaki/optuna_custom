import optuna
from optuna.samplers._tpe.sampler import TPESampler
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions, _BatchedCategoricalDistributions
import numpy as np
import matplotlib.pyplot as plt

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

configurations = {"A","B","C","D"}

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

config_results["A"] = np.array(all_runs)

for seed_n in range(roops):
    try_sampler = TPESampler(seed = seed_n,multivariate = False)
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

config_results["B"] = np.array(all_runs)

for seed_n in range(roops):
    try_sampler = TPESampler(seed = seed_n,multivariate = True)
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

config_results["C"] = np.array(all_runs)

for seed_n in range(roops):
    try_sampler = TPESampler(seed = seed_n,multivariate = True)
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

config_results["D"] = np.array(all_runs)

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