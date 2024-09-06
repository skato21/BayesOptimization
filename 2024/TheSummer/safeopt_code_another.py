from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union
import warnings

import numpy
import torch
from torch import Tensor
from packaging import version

from optuna import logging
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import proximal
import sys
import configparser
from epics import PV, caget, caput
import matplotlib.pyplot as plt



from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective import monte_carlo
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
)
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
import botorch.version
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition import qUpperConfidenceBound

if version.parse(botorch.version.version) < version.parse("0.8.0"):
    from botorch.fit import fit_gpytorch_model

    def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
        return SobolQMCNormalSampler(num_samples)

else:
    from botorch.fit import fit_gpytorch_mll

    def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
        return SobolQMCNormalSampler(torch.Size((num_samples,)))

from botorch.utils.multi_objective.box_decompositions import (
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import manual_seed
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


_logger = logging.get_logger(__name__)

def constrained_candidates_func(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    train_con: Optional[torch.Tensor],
    bounds: torch.Tensor,
    pending_x: torch.Tensor = None,
    acquisition_type_obj: str = "EI",
    acquisition_type_con: str = "EI",
    beta_obj: float = 2.0,
    beta_con: float = 2.0,
    constraint_threshold: float = 0.5,
    fig=None, ax=None,
) -> torch.Tensor:
    
    if train_con is not None:
        print("制約付きの候補点生成を行っています...")
    else:
        print("制約なしの候補点生成を行っています...")

    # 訓練データを正規化
    train_x = normalize(train_x, bounds=bounds)
    
    # 標準化された領域の境界
    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # 目的関数のためのGPモデルを作成
    model_obj = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.size(-1)))
    mll_obj = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    fit_gpytorch_mll(mll_obj)
    
    # 目的関数の獲得関数を選択
    if acquisition_type_obj == "EI":
        acqf_obj = ExpectedImprovement(model=model_obj, best_f=train_obj.max())
    elif acquisition_type_obj == "UCB":
        acqf_obj = UpperConfidenceBound(model=model_obj, beta=beta_obj)
    else:
        raise ValueError(f"Unknown acquisition type: {acquisition_type_obj}")
    
    candidates_obj, acq_value_obj = optimize_acqf(
        acq_function=acqf_obj,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    best_candidate_obj = unnormalize(candidates_obj.detach(), bounds=bounds)

    # 制約がある場合、制約のためのGPモデルを作成
    if train_con is not None:
        train_con = normalize(train_con, bounds=torch.tensor([[0.0], [1.0]]).to(train_con.device))
        model_con = SingleTaskGP(train_x, train_con, outcome_transform=Standardize(m=train_con.size(-1)))
        mll_con = ExactMarginalLogLikelihood(model_con.likelihood, model_con)
        fit_gpytorch_mll(mll_con)

        if acquisition_type_con == "EI":
            acqf_con = ExpectedImprovement(model=model_con, best_f=0.0)
        elif acquisition_type_con == "UCB":
            acqf_con = UpperConfidenceBound(model=model_con, beta=beta_con)
        else:
            raise ValueError(f"Unknown acquisition type for constraint: {acquisition_type_con}")
        
        candidates_con, acq_value_con = optimize_acqf(
            acq_function=acqf_con,
            bounds=standard_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
    else:
        acqf_con = None

    grid_size = 50
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    X, Y = torch.meshgrid(x, y)
    XY = torch.stack([X.ravel(), Y.ravel()], dim=-1).to(train_x.device)
    XY = XY.unsqueeze(1)
    
    with torch.no_grad():
        acq_values_obj = acqf_obj(XY).reshape(grid_size, grid_size).detach().cpu().numpy()
        if acq_values_obj.min() < 0:
            acq_values_obj = acq_values_obj - acq_values_obj.min()

        if acqf_con is not None:
            acq_values_con = acqf_con(XY).reshape(grid_size, grid_size).detach().cpu().numpy()

    def constrained_acqf(X):
        acq_value_obj = acqf_obj(X)

        if acqf_con is not None:
            acq_value_con = acqf_con(X)
            acqf_con_min = acq_value_con.min().item()
            acqf_con_max = acq_value_con.max().item()
            scaled_constraint_threshold = acqf_con_min + (acqf_con_max - acqf_con_min) * constraint_threshold
            constraint_satisfied = (acq_value_con - scaled_constraint_threshold <= 0).float()

            acq_value_obj = acq_value_obj - acq_value_obj.min().item()
            constrained_value = acq_value_obj * constraint_satisfied
            constrained_value = constrained_value - constrained_value.min().item()
            return constrained_value
        else:
            acq_value_obj = acq_value_obj - acq_value_obj.min().item()
            return acq_value_obj

    # constrained_acqf(X) の結果を計算して最大値を見つける
    constrained_acq_values = constrained_acqf(XY).reshape(grid_size, grid_size).detach().cpu().numpy()
    max_idx_constrained = constrained_acq_values.argmax()
    max_x_constrained, max_y_constrained = X.ravel()[max_idx_constrained], Y.ravel()[max_idx_constrained]

    # 制約付きの最適な候補点を返す
    best_candidate_constrained = torch.tensor([max_x_constrained, max_y_constrained], device=train_x.device).unsqueeze(0)
    best_candidate_constrained = unnormalize(best_candidate_constrained, bounds=bounds)

    # 定義域を元の座標に戻す
    X_plot, Y_plot = unnormalize(torch.stack([X, Y], dim=-1), bounds).unbind(-1)

    # プロットを更新
    if fig is not None:
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(24, 6))

    # 目的関数の獲得関数の最大値をプロット
    obj_acq_contour = ax[0].contourf(X_plot.numpy(), Y_plot.numpy(), acq_values_obj, levels=50, cmap="plasma")
    ax[0].scatter(unnormalize(train_x, bounds)[:, 0].cpu().numpy(), unnormalize(train_x, bounds)[:, 1].cpu().numpy(), color="red", marker="x", label="Training Data")
    ax[0].scatter(best_candidate_obj[:, 0].cpu().numpy(), best_candidate_obj[:, 1].cpu().numpy(), color="blue", marker="o", s=100, label="Max Value")
    ax[0].set_title("Objective Function (Acquisition)")
    ax[0].set_xlabel("X1")
    ax[0].set_ylabel("X2")
    fig.colorbar(obj_acq_contour, ax=ax[0])

    # 制約の獲得関数の最大値をプロット
    if acqf_con is not None:
        con_acq_contour = ax[1].contourf(X_plot.numpy(), Y_plot.numpy(), acq_values_con, levels=50, cmap="plasma")
        ax[1].scatter(unnormalize(train_x, bounds)[:, 0].cpu().numpy(), unnormalize(train_x, bounds)[:, 1].cpu().numpy(), color="red", marker="x", label="Training Data")
        #ax[1].scatter(best_candidate_constrained[:, 0].cpu().numpy(), best_candidate_constrained[:, 1].cpu().numpy(), color="blue", marker="o", s=100, label="Max Value")
        ax[1].set_title("Constraint Function (Acquisition)")
        ax[1].set_xlabel("X1")
        ax[1].set_ylabel("X2")
        fig.colorbar(con_acq_contour, ax=ax[1])

    # 制約付き獲得関数の最大値をプロット
    constrained_acq_contour = ax[2].contourf(X_plot.numpy(), Y_plot.numpy(), constrained_acq_values, levels=50, cmap="plasma")
    ax[2].scatter(unnormalize(train_x, bounds)[:, 0].cpu().numpy(), unnormalize(train_x, bounds)[:, 1].cpu().numpy(), color="red", marker="x", label="Training Data")
    ax[2].scatter(best_candidate_constrained[:, 0].cpu().numpy(), best_candidate_constrained[:, 1].cpu().numpy(), color="blue", marker="o", s=100, label="Max Value")
    ax[2].set_title("Constrained Acquisition Function")
    ax[2].set_xlabel("X1")
    ax[2].set_ylabel("X2")
    ax[2].legend()
    fig.colorbar(constrained_acq_contour, ax=ax[2])

    plt.tight_layout()
    plt.show()
    plt.pause(0.01)

    return best_candidate_constrained








@experimental_class("2.4.0")
class SafeOptSampler(BaseSampler):

    def __init__(
        self,
        x_name_list: list,
        x_min_max_list: list,
        x_weight_list: list,
        UCB_obj: bool,         # 目的関数のUCBフラグ
        EI_obj: bool,          # 目的関数のEIフラグ
        beta_obj: float,       # 目的関数のbeta値
        UCB_con: bool,         # 制約のUCBフラグ
        EI_con: bool,          # 制約のEIフラグ
        beta_con: float,       # 制約のbeta値
        constraint_threshold: float, #制約の閾値
        *,
        candidates_func: Optional[
            Callable[
                [
                    "torch.Tensor",  # train_x
                    "torch.Tensor",  # train_obj
                    Optional["torch.Tensor"],  # train_con
                    "torch.Tensor",  # bounds
                    Optional["torch.Tensor"],  # pending_x
                    str,            # acquisition_type_obj
                    str,            # acquisition_type_con
                    float,          # beta_obj
                    float,          # beta_con
                    float           # constraint_threshold
                ],
                "torch.Tensor",
            ]
        ] = None,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
        n_startup_trials: int = 10,
        consider_running_trials: bool = False,
        independent_sampler: Optional[BaseSampler] = None,
        seed: Optional[int] = None,
        device: Optional["torch.device"] = None,
    ):
        self.x_name_list = x_name_list
        self.x_min_max_list = x_min_max_list
        self.x_weight_list = x_weight_list
        self.UCB_obj = UCB_obj
        self.EI_obj = EI_obj
        self.beta_obj = beta_obj
        self.UCB_con = UCB_con
        self.EI_con = EI_con
        self.beta_con = beta_con
        self.constraint_threshold = constraint_threshold

        self._candidates_func = candidates_func or constrained_candidates_func
        self._constraints_func = constraints_func
        self._consider_running_trials = consider_running_trials
        self._independent_sampler = independent_sampler or RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._seed = seed

        self._study_id: Optional[int] = None
        self._search_space = IntersectionSearchSpace()
        self._device = device or torch.device("cpu")

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> Dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")

        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # built-in `candidates_func` cannot handle distributions that contain just a
                # single value, so we skip them. Note that the parameter values for such
                # distributions are sampled in `Trial`.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        assert isinstance(search_space, dict)

        if len(search_space) == 0:
            return {}

        completed_trials = study.get_trials(
            deepcopy=False, states=(TrialState.COMPLETE,)
        )

        n_completed_trials = len(completed_trials)
        con = None
        for trial_idx, trial in enumerate(completed_trials):
            constraints = trial.user_attrs.get("constraints")
            if constraints is not None:
                n_constraints = len(constraints)

                if con is None:
                    con = torch.full(
                        (n_completed_trials, n_constraints),
                        float('nan'),
                        dtype=torch.float64,  # torch.float64を指定
                        device=self._device
                    )

                con[trial_idx] = torch.tensor(constraints, dtype=torch.float64, device=self._device)
                
        #con = None  #debug

        running_trials = [
            t
            for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
            if t != trial
        ]
        trials = completed_trials + running_trials

        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            return {}

        trans = _SearchSpaceTransform(search_space)
        n_objectives = len(study.directions)
        values: Union[numpy.ndarray, torch.Tensor] = numpy.empty(
            (n_trials, n_objectives), dtype=numpy.float64  # dtypeをfloat64に変更
        )
        params: Union[numpy.ndarray, torch.Tensor]
        bounds: Union[numpy.ndarray, torch.Tensor] = trans.bounds.astype(numpy.float64)  # float64に変換
        params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)  # dtypeをfloat64に変更

        for trial_idx, trial in enumerate(trials):
            if trial.state == TrialState.COMPLETE:
                params[trial_idx] = trans.transform(trial.params).astype(numpy.float64)  # float64に変換
                assert len(study.directions) == len(trial.values)
                for obj_idx, (direction, value) in enumerate(
                    zip(study.directions, trial.values)
                ):
                    assert value is not None
                    if (
                        direction == StudyDirection.MINIMIZE
                    ):  # BoTorch always assumes maximization.
                        value *= -1
                    values[trial_idx, obj_idx] = value
            elif trial.state == TrialState.RUNNING:
                if all(p in trial.params for p in search_space):
                    params[trial_idx] = trans.transform(trial.params).astype(numpy.float64)  # float64に変換
                else:
                    params[trial_idx] = numpy.nan
            else:
                assert (
                    False
                ), "trial.state must be TrialState.COMPLETE or TrialState.RUNNING."

        values = torch.from_numpy(values).to(self._device)
        params = torch.from_numpy(params).to(self._device)
        bounds = torch.from_numpy(bounds).to(self._device)
        bounds.transpose_(0, 1)

        if self._candidates_func is None:
            self._candidates_func = constrained_candidates_func(
                n_objectives=n_objectives,
                has_constraint=con is not None,
                consider_running_trials=self._consider_running_trials,
            )

        completed_values = values[:n_completed_trials]
        completed_params = params[:n_completed_trials]
        if self._consider_running_trials:
            running_params = params[n_completed_trials:]
            running_params = running_params[~torch.isnan(running_params).any(dim=1)]
        else:
            running_params = None

        with manual_seed(self._seed):
            candidates = self._candidates_func(
                completed_params,          # train_x
                completed_values,          # train_obj
                con,                       # train_con
                bounds,                    # bounds
                pending_x=running_params,  # pending_x (optional)
                acquisition_type_obj="EI" if self.EI_obj else "UCB",  # acquisition_type_obj
                acquisition_type_con="EI" if self.EI_con else "UCB",  # acquisition_type_con
                beta_obj=self.beta_obj,    # 目的関数用のbeta値
                beta_con=self.beta_con,    # 制約用のbeta値
                constraint_threshold=self.constraint_threshold   # constraint_threshold (適宜変更可能)
            )
            if self._seed is not None:
                self._seed += 1

        if not isinstance(candidates, torch.Tensor):
            raise TypeError("Candidates must be a torch.Tensor.")
        if candidates.dim() == 2:
            if candidates.size(0) != 1:
                raise ValueError(
                    "Candidates batch optimization is not supported and the first dimension must "
                    "have size 1 if candidates is a two-dimensional tensor. Actual: "
                    f"{candidates.size()}."
                )
            # Batch size is one. Get rid of the batch dimension.
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError("Candidates must be one or two-dimensional.")
        if candidates.size(0) != bounds.size(1):
            raise ValueError(
                "Candidates size must match with the given bounds. Actual candidates: "
                f"{candidates.size(0)}, bounds: {bounds.size(1)}."
            )

        return trans.untransform(candidates.cpu().numpy())

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._seed is not None:
            self._seed = numpy.random.RandomState().randint(
                numpy.iinfo(numpy.int64).max
            )

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(
                self._constraints_func, study, trial, state
            )
        self._independent_sampler.after_trial(study, trial, state, values)


