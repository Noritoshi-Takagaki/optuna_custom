import optuna

def objective(trial):
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return x**2 + y**2

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler()
)
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best value: {study.best_value}")
