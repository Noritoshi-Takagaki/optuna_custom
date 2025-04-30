import optuna
import numpy as np
from scipy.stats import rv_discrete
from optuna.samplers import TPESampler

# 2つの事前分布を定義（例: 2値のカテゴリカル分布）
categories = [0, 1]
below_probabilities = [0.7, 0.3]  # good history (below)
above_probabilities = [0.4, 0.6]  # bad history (above)

below_distribution = rv_discrete(name='below_dist', values=(categories, below_probabilities))
above_distribution = rv_discrete(name='above_dist', values=(categories, above_probabilities))

# ParzenEstimator クラス（改造したもの）
class ParzenEstimator:
    def __init__(self):
        pass

    def _transform(self, samples_dict):
        return samples_dict['param1']

    def log_pdf_pr(self, samples_dict: dict[str, np.ndarray], user_distribution) -> np.ndarray:
        transformed_samples = self._transform(samples_dict)
        return user_distribution.logpmf(transformed_samples)  # rv_discreteの場合、logpmfを使用

    def _compute_acquisition_func(self, samples_dict, below_distribution, above_distribution, trial_count, beta):
        log_likelihoods_below = self.log_pdf_pr(samples_dict, below_distribution)
        log_likelihoods_above = self.log_pdf_pr(samples_dict, above_distribution)
        acq_func_vals = log_likelihoods_below - log_likelihoods_above
        return acq_func_vals

# 目的関数
def objective(trial):
    # パラメータの探索範囲を指定
    x = trial.suggest_float('x', -5, 5)
    y = trial.suggest_float('y', -5, 5)

    # ParzenEstimator インスタンス作成
    pe = ParzenEstimator()

    # トライアル数と定数βを指定（例: trial_count = 100, beta = 1.5）
    trial_count = 100
    beta = 1.5

    # 最適化で使うサンプル（例: 目的関数の評価用）
    samples_dict = {'param1': np.array([x, y])}

    # 獲得関数の計算
    acq_func_vals = pe._compute_acquisition_func(samples_dict, below_distribution, above_distribution, trial_count, beta)

    # 最適化の目的関数（獲得関数の値を最小化）
    return acq_func_vals.mean()  # 平均値を使って最小化する

# 最適化の実行
study = optuna.create_study(direction="minimize", sampler=TPESampler())
study.optimize(objective, n_trials=50)

# 最適化結果
print(f"Best objective value: {study.best_value}")
print(f"Best parameter: {study.best_params}")
