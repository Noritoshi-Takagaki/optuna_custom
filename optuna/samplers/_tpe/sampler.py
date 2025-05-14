from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import math
from typing import Any
from typing import cast
from typing import TYPE_CHECKING
import warnings

import numpy as np

from optuna import _deprecated
from optuna._experimental import warn_experimental_argument
from optuna._hypervolume import compute_hypervolume
from optuna._hypervolume.hssp import _solve_hssp
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._random import RandomSampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.search_space import IntersectionSearchSpace
from optuna.search_space.group_decomposed import _GroupDecomposedSearchSpace
from optuna.search_space.group_decomposed import _SearchSpaceGroup
from optuna.study._multi_objective import _fast_non_domination_rank
from optuna.study._multi_objective import _is_pareto_front
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


EPS = 1e-12
_logger = get_logger(__name__)


def default_gamma(x: int) -> int:
    return min(int(np.ceil(0.1 * x)), 25)


def hyperopt_default_gamma(x: int) -> int:
    return min(int(np.ceil(0.25 * np.sqrt(x))), 25)


def default_weights(x: int) -> np.ndarray:
    if x == 0:
        return np.asarray([])
    elif x < 25:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat], axis=0)


class TPESampler(BaseSampler):

    def __init__(
        self,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = False,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: Callable[[int], int] = default_gamma,
        weights: Callable[[int], np.ndarray] = default_weights,
        seed: int | None = None,
        *,
        multivariate: bool = False,
        group: bool = False,
        warn_independent_sampling: bool = True,
        constant_liar: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        categorical_distance_func: (
            dict[str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]] | None
        ) = None,
        below_distributions: dict[str, Any] | None = None,
        above_distributions: dict[str, Any] | None = None,
    ) -> None:
        if not consider_prior:
            msg = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
                name="`consider_prior`", d_ver="4.3.0", r_ver="6.0.0"
            )
            warnings.warn(msg, FutureWarning)

        self._parzen_estimator_parameters = _ParzenEstimatorParameters(
            consider_prior,
            prior_weight,
            consider_magic_clip,
            consider_endpoints,
            weights,
            multivariate,
            categorical_distance_func or {},
        )
        self._n_startup_trials = n_startup_trials
        self._n_ei_candidates = n_ei_candidates
        self._gamma = gamma

        self._warn_independent_sampling = warn_independent_sampling
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)

        self._multivariate = multivariate
        self._group = group
        self._group_decomposed_search_space: _GroupDecomposedSearchSpace | None = None
        self._search_space_group: _SearchSpaceGroup | None = None
        self._search_space = IntersectionSearchSpace(include_pruned=True)
        self._constant_liar = constant_liar
        self._constraints_func = constraints_func
        # NOTE(nabenabe0928): Users can overwrite _ParzenEstimator to customize the TPE behavior.
        self._parzen_estimator_cls = _ParzenEstimator

        self._below_distributions = below_distributions
        self._above_distributions = above_distributions

        if multivariate:
            warn_experimental_argument("multivariate")

        if group:
            if not multivariate:
                raise ValueError(
                    "``group`` option can only be enabled when ``multivariate`` is enabled."
                )
            warn_experimental_argument("group")
            self._group_decomposed_search_space = _GroupDecomposedSearchSpace(True)

        if constant_liar:
            warn_experimental_argument("constant_liar")

        if constraints_func is not None:
            warn_experimental_argument("constraints_func")

        if categorical_distance_func is not None:
            warn_experimental_argument("categorical_distance_func")

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        if not self._multivariate:
            return {}

        search_space: dict[str, BaseDistribution] = {}

        if self._group:
            assert self._group_decomposed_search_space is not None
            self._search_space_group = self._group_decomposed_search_space.calculate(study)
            for sub_space in self._search_space_group.search_spaces:
                # Sort keys because Python's string hashing is nondeterministic.
                for name, distribution in sorted(sub_space.items()):
                    if distribution.single():
                        continue
                    search_space[name] = distribution
            return search_space

        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if self._group:
            assert self._search_space_group is not None
            params = {}
            for sub_space in self._search_space_group.search_spaces:
                search_space = {}
                # Sort keys because Python's string hashing is nondeterministic.
                for name, distribution in sorted(sub_space.items()):
                    if not distribution.single():
                        search_space[name] = distribution
                params.update(self._sample_relative(study, trial, search_space))
            return params
        else:
            return self._sample_relative(study, trial, search_space)

    def _sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        # If the number of samples is insufficient, we run random trial.
        if len(trials) < self._n_startup_trials:
            return {}

        return self._sample(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        # If the number of samples is insufficient, we run random trial.
        if len(trials) < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        if self._warn_independent_sampling and self._multivariate:
            # Avoid independent warning at the first sampling of `param_name`.
            if any(param_name in trial.params for trial in trials):
                _logger.warning(
                    f"The parameter '{param_name}' in trial#{trial.number} is sampled "
                    "independently instead of being sampled by multivariate TPE sampler. "
                    "(optimization performance may be degraded). "
                    "You can suppress this warning by setting `warn_independent_sampling` "
                    "to `False` in the constructor of `TPESampler`, "
                    "if this independent sampling is intended behavior."
                )

        return self._sample(study, trial, {param_name: param_distribution})[param_name]

    def _get_internal_repr(
        self, trials: list[FrozenTrial], search_space: dict[str, BaseDistribution]
    ) -> dict[str, np.ndarray]:
        values: dict[str, list[float]] = {param_name: [] for param_name in search_space}
        for trial in trials:
            if all((param_name in trial.params) for param_name in search_space):
                for param_name in search_space:
                    param = trial.params[param_name]
                    distribution = trial.distributions[param_name]
                    values[param_name].append(distribution.to_internal_repr(param))
        return {k: np.asarray(v) for k, v in values.items()}

    def _sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if self._constant_liar:
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=states, use_cache=use_cache)

        # We divide data into below and above.
        n = sum(trial.state != TrialState.RUNNING for trial in trials)  # Ignore running trials.
        below_trials, above_trials = _split_trials(
            study,
            trials,
            self._gamma(n),
            self._constraints_func is not None,
        )

        mpe_below = self._build_parzen_estimator(
            study, search_space, below_trials, handle_below=True
        )
        mpe_above = self._build_parzen_estimator(
            study, search_space, above_trials, handle_below=False
        )

        samples_below = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
        acq_func_vals = self._compute_acquisition_func(samples_below, mpe_below, mpe_above, study)
        ret = TPESampler._compare(samples_below, acq_func_vals)

        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret

    def _build_parzen_estimator(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        trials: list[FrozenTrial],
        handle_below: bool,
    ) -> _ParzenEstimator:
        observations = self._get_internal_repr(trials, search_space)
        if handle_below and study._is_multi_objective():
            param_mask_below = []
            for trial in trials:
                param_mask_below.append(
                    all((param_name in trial.params) for param_name in search_space)
                )
            weights_below = _calculate_weights_below_for_multi_objective(
                study, trials, self._constraints_func
            )[param_mask_below]
            assert np.isfinite(weights_below).all()
            mpe = self._parzen_estimator_cls(
                observations, search_space, self._parzen_estimator_parameters, weights_below
            )
        else:
            mpe = self._parzen_estimator_cls(
                observations, search_space, self._parzen_estimator_parameters
            )

        if not isinstance(mpe, _ParzenEstimator):
            raise RuntimeError("_parzen_estimator_cls must override _ParzenEstimator.")

        return mpe

    def _compute_acquisition_func(
        self,
        samples: dict[str, np.ndarray],
        mpe_below: _ParzenEstimator,
        mpe_above: _ParzenEstimator,
        study,
        beta: float = 10  # ユーザーが指定するβの定数
    ) -> np.ndarray:
        t = len(study.trials)
        scale_factor = t/beta
        log_likelihoods_below = mpe_below.log_pdf(samples)
        log_likelihoods_above = mpe_above.log_pdf(samples)
        #print(list(samples)) #
        #print(log_likelihoods_below) #
        #print(log_likelihoods_above) #
        if self._below_distributions is None or self._above_distributions is None:
            acq_func_vals = log_likelihoods_below - log_likelihoods_above
        elif self._multivariate:
            log_likelihoods_below_given_distributions_list = []
            log_likelihoods_above_given_distributions_list = []
            acq_func_vals = scale_factor * log_likelihoods_below - scale_factor * log_likelihoods_above
            #print(range(len(list(samples)))) #
            for i in range(len(list(samples))): #他変量分布が積の形で表されると仮定
                log_likelihoods_below_given_distributions_list.append(mpe_below.log_pdf_given_distributions(samples, self._below_distributions[list(samples)[i]]))
                log_likelihoods_above_given_distributions_list.append(mpe_above.log_pdf_given_distributions(samples, self._above_distributions[list(samples)[i]]))
                acq_func_vals += log_likelihoods_below_given_distributions_list[i]
                acq_func_vals -= log_likelihoods_above_given_distributions_list[i]
                #print(acq_func_vals) #
                #print(list(samples)[i]) #
                #print(log_likelihoods_below_given_distributions_list[i]) #
                #print(log_likelihoods_above_given_distributions_list[i]) #
        else:
            log_likelihoods_below_given_distributions = mpe_below.log_pdf_given_distributions(samples, self._below_distributions[list(samples)[0]])
            log_likelihoods_above_given_distributions = mpe_above.log_pdf_given_distributions(samples, self._above_distributions[list(samples)[0]])
            #print(log_likelihoods_below_given_distributions) #
            #print(log_likelihoods_above_given_distributions) #
            acq_func_vals = log_likelihoods_below_given_distributions + scale_factor * log_likelihoods_below - log_likelihoods_above_given_distributions - scale_factor * log_likelihoods_above     
        return acq_func_vals

    @classmethod
    def _compare(
        cls, samples: dict[str, np.ndarray], acquisition_func_vals: np.ndarray
    ) -> dict[str, int | float]:
        sample_size = next(iter(samples.values())).size
        if sample_size == 0:
            raise ValueError(f"The size of `samples` must be positive, but got {sample_size}.")

        if sample_size != acquisition_func_vals.size:
            raise ValueError(
                "The sizes of `samples` and `acquisition_func_vals` must be same, but got "
                "(samples.size, acquisition_func_vals.size) = "
                f"({sample_size}, {acquisition_func_vals.size})."
            )

        best_idx = np.argmax(acquisition_func_vals)
        return {k: v[best_idx].item() for k, v in samples.items()}

    @staticmethod
    def hyperopt_parameters() -> dict[str, Any]:
        """Return the the default parameters of hyperopt (v0.1.2).

        :class:`~optuna.samplers.TPESampler` can be instantiated with the parameters returned
        by this method.

        Example:

            Create a :class:`~optuna.samplers.TPESampler` instance with the default
            parameters of `hyperopt <https://github.com/hyperopt/hyperopt/tree/0.1.2>`__.

            .. testcode::

                import optuna
                from optuna.samplers import TPESampler


                def objective(trial):
                    x = trial.suggest_float("x", -10, 10)
                    return x**2


                sampler = TPESampler(**TPESampler.hyperopt_parameters())
                study = optuna.create_study(sampler=sampler)
                study.optimize(objective, n_trials=10)

        Returns:
            A dictionary containing the default parameters of hyperopt.

        """

        return {
            "consider_prior": True,
            "prior_weight": 1.0,
            "consider_magic_clip": True,
            "consider_endpoints": False,
            "n_startup_trials": 20,
            "n_ei_candidates": 24,
            "gamma": hyperopt_default_gamma,
            "weights": default_weights,
        }

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._random_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._random_sampler.after_trial(study, trial, state, values)


def _get_reference_point(loss_vals: np.ndarray) -> np.ndarray:
    worst_point = np.max(loss_vals, axis=0)
    reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
    reference_point[reference_point == 0] = EPS
    return reference_point


def _split_trials(
    study: Study, trials: list[FrozenTrial], n_below: int, constraints_enabled: bool
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    complete_trials = []
    pruned_trials = []
    running_trials = []
    infeasible_trials = []

    for trial in trials:
        if trial.state == TrialState.RUNNING:
            # We should check if the trial is RUNNING before the feasibility check
            # because its constraint values have not yet been set.
            running_trials.append(trial)
        elif constraints_enabled and _get_infeasible_trial_score(trial) > 0:
            infeasible_trials.append(trial)
        elif trial.state == TrialState.COMPLETE:
            complete_trials.append(trial)
        elif trial.state == TrialState.PRUNED:
            pruned_trials.append(trial)
        else:
            assert False

    # We divide data into below and above.
    below_complete, above_complete = _split_complete_trials(complete_trials, study, n_below)
    # This ensures `n_below` is non-negative to prevent unexpected trial splits.
    n_below = max(0, n_below - len(below_complete))
    below_pruned, above_pruned = _split_pruned_trials(pruned_trials, study, n_below)
    # This ensures `n_below` is non-negative to prevent unexpected trial splits.
    n_below = max(0, n_below - len(below_pruned))
    below_infeasible, above_infeasible = _split_infeasible_trials(infeasible_trials, n_below)

    below_trials = below_complete + below_pruned + below_infeasible
    above_trials = above_complete + above_pruned + above_infeasible + running_trials
    below_trials.sort(key=lambda trial: trial.number)
    above_trials.sort(key=lambda trial: trial.number)

    return below_trials, above_trials


def _split_complete_trials(
    trials: Sequence[FrozenTrial], study: Study, n_below: int
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    n_below = min(n_below, len(trials))
    if len(study.directions) <= 1:
        return _split_complete_trials_single_objective(trials, study, n_below)
    else:
        return _split_complete_trials_multi_objective(trials, study, n_below)


def _split_complete_trials_single_objective(
    trials: Sequence[FrozenTrial], study: Study, n_below: int
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    if study.direction == StudyDirection.MINIMIZE:
        sorted_trials = sorted(trials, key=lambda trial: cast(float, trial.value))
    else:
        sorted_trials = sorted(trials, key=lambda trial: cast(float, trial.value), reverse=True)
    return sorted_trials[:n_below], sorted_trials[n_below:]


def _split_complete_trials_multi_objective(
    trials: Sequence[FrozenTrial], study: Study, n_below: int
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    if n_below == 0:
        return [], list(trials)
    elif n_below == len(trials):
        return list(trials), []

    assert 0 < n_below < len(trials)
    lvals = np.array([trial.values for trial in trials])
    lvals *= np.array([-1.0 if d == StudyDirection.MAXIMIZE else 1.0 for d in study.directions])
    nondomination_ranks = _fast_non_domination_rank(lvals, n_below=n_below)
    ranks, rank_counts = np.unique(nondomination_ranks, return_counts=True)
    last_rank_before_tiebreak = int(np.max(ranks[np.cumsum(rank_counts) <= n_below], initial=-1))
    assert all(ranks[: last_rank_before_tiebreak + 1] == np.arange(last_rank_before_tiebreak + 1))
    indices = np.arange(len(trials))
    indices_below = indices[nondomination_ranks <= last_rank_before_tiebreak]

    if indices_below.size < n_below:  # Tie-break with Hypervolume subset selection problem (HSSP).
        assert ranks[last_rank_before_tiebreak + 1] == last_rank_before_tiebreak + 1
        need_tiebreak = nondomination_ranks == last_rank_before_tiebreak + 1
        rank_i_lvals = lvals[need_tiebreak]
        subset_size = n_below - indices_below.size
        selected_indices = _solve_hssp(
            rank_i_lvals, indices[need_tiebreak], subset_size, _get_reference_point(rank_i_lvals)
        )
        indices_below = np.append(indices_below, selected_indices)

    below_indices_set = set(cast(list, indices_below.tolist()))
    below_trials = [trials[i] for i in range(len(trials)) if i in below_indices_set]
    above_trials = [trials[i] for i in range(len(trials)) if i not in below_indices_set]
    return below_trials, above_trials


def _get_pruned_trial_score(trial: FrozenTrial, study: Study) -> tuple[float, float]:
    if len(trial.intermediate_values) > 0:
        step, intermediate_value = max(trial.intermediate_values.items())
        if math.isnan(intermediate_value):
            return -step, float("inf")
        elif study.direction == StudyDirection.MINIMIZE:
            return -step, intermediate_value
        else:
            return -step, -intermediate_value
    else:
        return 1, 0.0


def _split_pruned_trials(
    trials: Sequence[FrozenTrial], study: Study, n_below: int
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    n_below = min(n_below, len(trials))
    sorted_trials = sorted(trials, key=lambda trial: _get_pruned_trial_score(trial, study))
    return sorted_trials[:n_below], sorted_trials[n_below:]


def _get_infeasible_trial_score(trial: FrozenTrial) -> float:
    constraint = trial.system_attrs.get(_CONSTRAINTS_KEY)
    if constraint is None:
        warnings.warn(
            f"Trial {trial.number} does not have constraint values."
            " It will be treated as a lower priority than other trials."
        )
        return float("inf")
    else:
        # Violation values of infeasible dimensions are summed up.
        return sum(v for v in constraint if v > 0)


def _split_infeasible_trials(
    trials: Sequence[FrozenTrial], n_below: int
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    n_below = min(n_below, len(trials))
    sorted_trials = sorted(trials, key=_get_infeasible_trial_score)
    return sorted_trials[:n_below], sorted_trials[n_below:]


def _calculate_weights_below_for_multi_objective(
    study: Study,
    below_trials: list[FrozenTrial],
    constraints_func: Callable[[FrozenTrial], Sequence[float]] | None,
) -> np.ndarray:
    def _feasible(trial: FrozenTrial) -> bool:
        return constraints_func is None or all(c <= 0 for c in constraints_func(trial))

    is_feasible = np.asarray([_feasible(t) for t in below_trials])
    weights_below = np.where(is_feasible, 1.0, EPS)  # Assign EPS to infeasible trials.
    n_below_feasible = np.count_nonzero(is_feasible)
    if n_below_feasible <= 1:
        return weights_below

    lvals = np.asarray([t.values for t in below_trials])[is_feasible]
    lvals *= np.array([-1.0 if d == StudyDirection.MAXIMIZE else 1.0 for d in study.directions])
    ref_point = _get_reference_point(lvals)
    on_front = _is_pareto_front(lvals, assume_unique_lexsorted=False)
    pareto_sols = lvals[on_front]
    hv = compute_hypervolume(pareto_sols, ref_point, assume_pareto=True)
    if np.isinf(hv):
        # TODO(nabenabe): Assign EPS to non-Pareto solutions, and
        # solutions with finite contrib if hv is inf. Ref: PR#5813.
        return weights_below

    loo_mat = ~np.eye(pareto_sols.shape[0], dtype=bool)  # Leave-one-out bool matrix.
    contribs = np.zeros(n_below_feasible, dtype=float)
    contribs[on_front] = hv - np.array(
        [compute_hypervolume(pareto_sols[loo], ref_point) for loo in loo_mat]
    )
    weights_below[is_feasible] = np.maximum(contribs / max(np.max(contribs), EPS), EPS)
    return weights_below
