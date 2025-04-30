import optuna
import numpy as np
from optuna.samplers import TPESampler
from optuna import Trial

# カスタム TPE サンプラーの作成
class CustomTPESampler(TPESampler):
    def __init__(self, prior_mu_sigma_g=None, prior_mu_sigma_b=None, **kwargs):
        super().__init__(**kwargs)
        self.prior_mu_sigma_g = prior_mu_sigma_g
        self.prior_mu_sigma_b = prior_mu_sigma_b

    def _call_weights_func(self, weights_func: callable, n: int) -> np.ndarray:
        # weights_func は 'uniform' などの文字列ではなく、実際の関数として処理する必要がある
        if isinstance(weights_func, str):
            if weights_func == 'uniform':
                return np.ones(n)
            else:
                raise ValueError(f"Unsupported weight function: {weights_func}")
        else:
            return np.array(weights_func(n))[:n]

    def _build_parzen_estimator(self, study, is_good, n_trials, trials, distributions):
        # TPEの履歴から「良い」「悪い」履歴を分ける
        # is_good = True の場合、良い履歴、False の場合は悪い履歴
        if is_good:
            print("Building Parzen estimator for good trials.")
        else:
            print("Building Parzen estimator for bad trials.")
        
        return super()._build_parzen_estimator(
            study, is_good, n_trials, trials, distributions
        )

# 目的関数の定義
def objective(trial: Trial):
    # サンプルするパラメータの設定
    param1 = trial.suggest_int('param1', 0, 10)
    param2 = trial.suggest_float('param2', 0.0, 1.0)
    
    # 目的関数 (例: パラメータの偏差の二乗和)
    return (param1 - 5) ** 2 + (param2 - 0.5) ** 2

# カスタム TPE サンプラーの設定
sampler = CustomTPESampler(
    prior_mu_sigma_g=(0.0, 1.0),  # グッドヒストリー用の事前分布
    prior_mu_sigma_b=(0.0, 1.0)   # バッドヒストリー用の事前分布
)

# 最適化の実行
study = optuna.create_study(sampler=sampler, direction='minimize')
study.optimize(objective, n_trials=100)

# 最適化結果の表示
print(f"Best trial value: {study.best_trial.value}")
print(f"Best trial parameters: {study.best_trial.params}")
