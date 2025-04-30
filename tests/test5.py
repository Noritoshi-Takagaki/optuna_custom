import optuna
from optuna.samplers import TPESampler
import numpy as np

# 目的関数
def objective(trial):
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return x**2 + y**2

# CustomTPESamplerを使って最適化
study = optuna.create_study(sampler=TPESampler())
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best value: {study.best_value}")