"""Construct supported regressors from validated declarative settings."""

from importlib import import_module

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.config.base_estimator import GaussianConfig, ModelConfig


def uses_numeric_scaling(base: ModelConfig) -> bool:
    """Scale numeric columns for linear, kernel, and nearest-neighbor models."""
    return base.estimator in {"ridge", "elastic_net", "gaussian_process", "knn", "svr"}


def _optional_library(base: ModelConfig) -> object:
    """Load only the optional booster required by the selected model."""
    try:
        return import_module(base.estimator)
    except (ImportError, OSError) as error:
        raise ValueError(f"Model {base.name!r} selects estimator={base.estimator!r}; expected its declared CPU "
                         f"dependency to be installed and loadable: {error}") from error


def _gaussian_estimator(base: GaussianConfig, seed: int) -> RegressorMixin:
    """Turn the kernel declaration into a new sklearn kernel for this fit."""
    from src.model.iterative import TracedGaussianProcess

    parameters = base.parameters.model_dump(exclude={"kernel"})
    settings = base.parameters.kernel
    kernel_fields = settings.model_dump(exclude={"kind"})
    kernel = RBF(**kernel_fields) if settings.kind == "rbf" else Matern(**kernel_fields)
    return TracedGaussianProcess(**parameters, kernel=kernel, random_state=seed)


def _booster(base: ModelConfig, seed: int) -> RegressorMixin:
    """Pin CPU-only, single-threaded library settings without early stopping."""
    library = _optional_library(base)
    parameters = base.parameters.model_dump()
    if base.estimator == "lightgbm":
        return library.LGBMRegressor(**parameters, random_state=seed % (2**31), n_jobs=1, device_type="cpu",
                                    objective="regression", deterministic=True, force_col_wise=True, verbosity=-1)
    if base.estimator == "catboost":
        return library.CatBoostRegressor(**parameters, random_seed=seed % (2**31), thread_count=1, task_type="CPU",
                                        loss_function="RMSE", allow_writing_files=False, use_best_model=False,
                                        allow_const_label=True, verbose=False)
    return library.XGBRegressor(**parameters, random_state=seed % (2**31), n_jobs=1, device="cpu", tree_method="hist",
                                objective="reg:squarederror", eval_metric="rmse", verbosity=0)


def make_estimator(base: ModelConfig, seed: int) -> RegressorMixin:
    """Build a fresh named estimator for one permitted training scope."""
    if base.estimator == "gaussian_process":
        return _gaussian_estimator(base, seed)
    parameters = base.parameters.model_dump()
    if base.estimator == "ridge":
        return Ridge(**parameters, random_state=seed)
    if base.estimator == "random_forest":
        return RandomForestRegressor(**parameters, random_state=seed)
    if base.estimator == "elastic_net":
        from src.model.iterative import TracedElasticNet

        return TracedElasticNet(**parameters, warm_start=False, random_state=seed)
    if base.estimator == "knn":
        return KNeighborsRegressor(**parameters, n_jobs=1)
    if base.estimator == "svr":
        return SVR(**parameters)
    return _booster(base, seed)


def effective_parameters(base: ModelConfig, estimator: RegressorMixin) -> dict:
    """Record declared settings and scalar GridSearchCV winner values as JSON data."""
    configured = base.parameters.model_dump(mode="json")
    fitted = estimator.get_params(deep=False)
    for name in configured:
        if base.estimator != "gaussian_process" or name != "kernel":
            configured[name] = fitted[name]
    if base.estimator == "gaussian_process" and hasattr(estimator, "kernel_"):
        configured["fitted_kernel"] = dict(configured["kernel"], length_scale=float(estimator.kernel_.length_scale))
    return configured
