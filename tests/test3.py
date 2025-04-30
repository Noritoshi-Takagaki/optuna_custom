import optuna
from scipy.stats import norm
from optuna.samplers._tpe.sampler import TPESampler

# ユーザー指定の分布
prior_distribution_below = norm(loc=0, scale=1)
prior_distribution_above = norm(loc=2, scale=1)

# TPESampler を使用して分布を渡す
sampler = TPESampler(
    prior_distribution_below=prior_distribution_below,
    prior_distribution_above=prior_distribution_above
)

# 目標関数
def objective(trial):
    x = trial.suggest_float('x', -5, 5)
    y = trial.suggest_float('y', -5, 5)
    return x**2 + y**2

# 最適化実行
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)

print("Best value:", study.best_value)
print("Best parameters:", study.best_params)
