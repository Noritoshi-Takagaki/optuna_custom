import pytest
import optuna
import numpy as np
from unittest.mock import MagicMock
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters, _MixtureOfProductDistribution

def uniform_weights(n):
    return np.ones(n)  # n個の均等な重みを返す関数

# mock_search_space フィクスチャ
@pytest.fixture
def mock_search_space():
    return {
        'param1': optuna.distributions.IntDistribution(0, 10),
        'param2': optuna.distributions.FloatDistribution(0.0, 1.0),
    }

# 他のモックフィクスチャも必要に応じて追加
@pytest.fixture
def mock_observations():
    return {
        'param1': np.array([1, 2, 3]),
        'param2': np.array([0.1, 0.2, 0.3]),
    }

@pytest.fixture
def mock_parameters():
    return _ParzenEstimatorParameters(
        consider_prior=True, 
        prior_weight=1.0, 
        consider_magic_clip=True, 
        consider_endpoints=True, 
        weights=uniform_weights,  # 'uniform' ではなく、関数を渡す
        multivariate=False, 
        categorical_distance_func=None
    )

@pytest.fixture
def mock_parameters():
    return _ParzenEstimatorParameters(
        consider_prior=True,
        prior_weight=1.0,
        weights="uniform",
        consider_magic_clip=True,
        consider_endpoints=True,
        multivariate=False,
        categorical_distance_func=None
    )

@pytest.fixture
def mock_prior_log_prob():
    return {"param1": np.log(0.3), "param2": np.log(0.7)}

@pytest.fixture
def mock_mixture_distribution():
    mock = MagicMock(spec=_MixtureOfProductDistribution)
    mock.log_pdf.return_value = np.array([0.5])
    return mock

@pytest.fixture
def parzen_estimator(mock_observations, mock_search_space, mock_parameters, mock_prior_log_prob):
    return _ParzenEstimator(
        observations=mock_observations,
        search_space=mock_search_space,
        parameters=mock_parameters
    )

def test_prior_log_prob_storage(parzen_estimator, mock_prior_log_prob):
    # 初期化時にprior_log_probが正しく格納されるかを確認
    assert parzen_estimator._prior_log_prob == mock_prior_log_prob

def test_acquisition_func_with_prior(parzen_estimator, mock_mixture_distribution):
    # mock_mixture_distributionをmpe_belowとmpe_aboveにセット
    mpe_below = MagicMock(spec=_ParzenEstimator)
    mpe_above = MagicMock(spec=_ParzenEstimator)
    
    # それぞれのprior_log_probを設定
    mpe_below._prior_log_prob = {"param1": np.log(0.3), "param2": np.log(0.7)}
    mpe_above._prior_log_prob = {"param1": np.log(0.5), "param2": np.log(0.2)}
    
    # log_pdfをモックする
    mpe_below.log_pdf.return_value = np.array([0.6])
    mpe_above.log_pdf.return_value = np.array([0.4])
    
    # _compute_acquisition_funcを呼び出し、獲得関数の計算をテスト
    t = 10  # トライアル数
    beta = 1.5  # 定数β
    samples = {"param1": np.array([0.5]), "param2": np.array([0.2])}
    
    acq_func_vals = parzen_estimator._compute_acquisition_func(samples, mpe_below, mpe_above, t, beta)
    
    # 期待される獲得関数の値を計算
    expected_acq = (
        np.log(0.3) + (t / beta) * 0.6  # Pbの項
        - np.log(0.2) - (t / beta) * 0.4  # Paの項
    )
    
    # 計算結果が期待通りかをチェック
    assert np.allclose(acq_func_vals, expected_acq)

