from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
    make_column_transformer,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge


def get_column_transformer() -> ColumnTransformer:
    return make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(dtype=int), make_column_selector(dtype_exclude=np.number)),
    )


def get_feature_selector(
    feature_selector: str,
    pca_components: Optional[int] = None,
    k: int = 10,
) -> Union[PCA, SelectKBest]:

    feature_selectors = {"pca": PCA(pca_components), "k_best": SelectKBest(k=k)}
    return feature_selectors[feature_selector]


def get_preprocessor(
    column_transformer: ColumnTransformer, feature_selector: Union[PCA, SelectKBest]
) -> Pipeline:
    return Pipeline(
        [
            ("column_transformer", column_transformer),
            ("feature_selector", feature_selector),
        ]
    )


def update_dict_keys(d: dict, prefix: bool = False, replace: bool = False):
    if prefix:
        new_dict = {"model__" + key: value for key, value in d.items()}
    if replace:
        new_dict = {
            key.replace("model__", "model__regressor__"): value
            for key, value in d.items()
        }
    return new_dict


@dataclass
class ModelContainer:
    model: Any
    preprocessor: Union[ColumnTransformer, Pipeline]
    param_grid: Optional[dict] = None
    log_y: bool = False

    def __post_init__(self):
        # allow param_grid key as 'alpha' instead of 'model__alpha'
        if self.param_grid is not None:
            self.param_grid = update_dict_keys(self.param_grid, prefix=True)

        # add extra step of log-transforming price
        if self.log_y:
            self.model = TransformedTargetRegressor(
                self.model, func=np.log, inverse_func=np.exp
            )

            # adjust names in param_grid after log transformation
            # new key name is e.g. model__regressor__alpha
            if self.param_grid is not None:
                self.param_grid = update_dict_keys(self.param_grid, replace=True)

        self.pipeline = Pipeline(
            [("preprocessor", self.preprocessor), ("model", self.model)]
        )


@dataclass
class ResultContainer:
    model_names: list[str] = field(default_factory=list)
    train_mae_list: list[float] = field(default_factory=list)
    val_mae_list: list[float] = field(default_factory=list)
    train_r2_list: list[float] = field(default_factory=list)
    val_r2_list: list[float] = field(default_factory=list)
    train_mse_list: list[float] = field(default_factory=list)
    val_mse_list: list[float] = field(default_factory=list)
    hyperparam_keys: list[str] = field(default_factory=list)
    hyperparam_values: list[float] = field(default_factory=list)
    num_features: list[float] = field(default_factory=list)
    feature_selector: list[float] = field(default_factory=list)
    log_y: list[bool] = field(default_factory=list)

    def append(
        self,
        train_mae: float,
        val_mae: float,
        train_r2: float,
        val_r2: float,
        train_mse: float,
        val_mse: float,
        hyperparam_key: Optional[Union[str, list[str]]] = None,
        hyperparam_value: Optional[Union[str, float, list[float]]] = None,
        num_features: Optional[int] = None,
    ):
        self.train_mae_list.append(train_mae)
        self.val_mae_list.append(val_mae)
        self.train_r2_list.append(train_r2)
        self.val_r2_list.append(val_r2)
        self.train_mse_list.append(train_mse)
        self.val_mse_list.append(val_mse)

        if hyperparam_key is not None:
            self.hyperparam_keys.append(hyperparam_key)
        if hyperparam_value is not None:
            self.hyperparam_values.append(hyperparam_value)
        if num_features is not None:
            self.num_features.append(num_features)

    def display_df(self) -> pd.DataFrame:
        metrics_df = pd.DataFrame(
            {
                "mae_train": self.train_mae_list,
                "mae_val": self.val_mae_list,
                "r2_train": self.train_r2_list,
                "r2_val": self.val_r2_list,
                "mse_train": self.train_mse_list,
                "mse_val": self.val_mse_list,
                "hyperparam_keys": self.hyperparam_keys,
                "hyperparam_values": self.hyperparam_values,
                "num_features": self.num_features,
                "feature_selector": self.feature_selector,
                "log_y": self.log_y,
            },
            index=self.model_names,
        )

        return metrics_df.sort_values("mae_val")


def get_models(
    preprocessor: Union[ColumnTransformer, Pipeline],
    models: Optional[list[str]] = None,
    random_state: Optional[int] = None,
    log_y: bool = False,
) -> list[ModelContainer]:

    linear = ModelContainer(LinearRegression(), preprocessor, log_y=log_y)
    lasso = ModelContainer(
        Lasso(random_state=random_state),
        preprocessor,
        {"alpha": np.arange(1, 50)},
        log_y=log_y,
    )

    ridge = ModelContainer(
        Ridge(random_state=random_state),
        preprocessor,
        {"alpha": np.arange(10, 1000, 10)},
        log_y=log_y,
    )

    random_forest = ModelContainer(
        RandomForestRegressor(random_state=random_state),
        preprocessor,
        {
            "max_depth": np.arange(1, 10),
            "min_samples_leaf": np.arange(1, 10),
            "n_estimators": np.arange(1, 10),
        },
        log_y=log_y,
    )

    gradient_boosting = ModelContainer(
        GradientBoostingRegressor(random_state=random_state),
        preprocessor,
        {
            "learning_rate": np.arange(0.1, 1, 0.1),
            "max_depth": np.arange(1, 10),
            "min_samples_leaf": np.arange(1, 10),
            "n_estimators": np.arange(1, 10),
            "subsample": np.arange(0.01, 0.2, 0.02),
        },
        log_y=log_y,
    )

    ada_boost = ModelContainer(
        AdaBoostRegressor(random_state=random_state),
        preprocessor,
        {
            "learning_rate": np.arange(1, 5),
            "n_estimators": np.arange(2, 20, 2),
        },
        log_y=log_y,
    )

    bagging = ModelContainer(
        BaggingRegressor(random_state=random_state),
        preprocessor,
        {
            "max_features": np.arange(0.1, 1, 0.1),
            "max_samples": np.arange(0.01, 0.1, 0.01),
            "n_estimators": np.arange(10, 50, 10),
        },
        log_y=log_y,
    )

    model_choice = {
        "linear": linear,
        "lasso": lasso,
        "ridge": ridge,
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting,
        "ada_boost": ada_boost,
        "bagging": bagging,
    }

    if models is None:
        return list(model_choice.values())

    return [model_choice[model] for model in models]


def get_model_name(model: ModelContainer, log_y: bool):
    if log_y:
        return model.model.regressor.__class__.__name__
    else:
        return model.model.__class__.__name__


def get_feature_selector_name(model_container: ModelContainer) -> Optional[str]:
    try:
        return model_container.preprocessor.named_steps[
            "feature_selector"
        ].__class__.__name__
    # if no feature_selector was used
    except (KeyError, AttributeError):
        return None


def fit_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[ModelContainer],
    result_container: ResultContainer,
    n_folds: int = 5,
    n_iter: int = 10,
    random_state: Optional[int] = None,
    log_y: bool = False,
) -> ResultContainer:
    start = perf_counter()

    for model in models:
        model_name = get_model_name(model, log_y)
        print("Fitting", model_name)
        result_container.model_names.append(model_name)
        result_container.feature_selector.append(get_feature_selector_name(model))
        result_container.log_y.append(log_y)

        scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

        # if model does not have hyperparameters
        if model.param_grid is None:
            scores = cross_validate(
                model.pipeline,
                X,
                y,
                cv=n_folds,
                scoring=scoring,
                return_train_score=True,
                return_estimator=True,
            )

            train_mae = -np.mean(scores["train_neg_mean_absolute_error"])
            val_mae = -np.mean(scores["test_neg_mean_absolute_error"])
            train_r2 = np.mean(scores["train_r2"])
            val_r2 = np.mean(scores["test_r2"])
            train_mse = -np.mean(scores["train_neg_mean_squared_error"])
            val_mse = -np.mean(scores["test_neg_mean_squared_error"])

            # specify as strings to be compatible with default behaviour for None values of ResultContainer.append()
            hyperparam_key = "None"
            hyperparam_value = "None"

            # accessing attributes depends on log_y
            estimator = scores["estimator"][0]["model"]
            num_features = (
                estimator.regressor_.n_features_in_
                if log_y
                else estimator.n_features_in_
            )

        # if model has hyperparameters
        else:
            cv = RandomizedSearchCV(
                estimator=model.pipeline,
                param_distributions=model.param_grid,
                cv=n_folds,
                n_iter=n_iter,
                scoring=scoring,
                refit="neg_mean_absolute_error",
                return_train_score=True,
                random_state=random_state,
            )

            cv.fit(X, y)

            best_index = cv.cv_results_["params"].index(cv.best_params_)

            # for some reason, only negative mean squared/absolute error are available metrics
            train_mae = -cv.cv_results_["mean_train_neg_mean_absolute_error"][
                best_index
            ]
            val_mae = -cv.cv_results_["mean_test_neg_mean_absolute_error"][best_index]
            train_r2 = cv.cv_results_["mean_train_r2"][best_index]
            val_r2 = cv.cv_results_["mean_test_r2"][best_index]
            train_mse = -cv.cv_results_["mean_train_neg_mean_squared_error"][best_index]
            val_mse = -cv.cv_results_["mean_test_neg_mean_squared_error"][best_index]

            split_index = 2 if log_y else 1
            hyperparam_key = [key.split("__")[split_index] for key in cv.best_params_]
            hyperparam_value = [value for value in cv.best_params_.values()]
            num_features = cv.best_estimator_.named_steps["model"].n_features_in_

            # display scalars in DataFrame instead of one-element lists
            if len(model.param_grid) == 1:
                hyperparam_key = hyperparam_key[0]
                hyperparam_value = hyperparam_value[0]

        result_container.append(
            train_mae,
            val_mae,
            train_r2,
            val_r2,
            train_mse,
            val_mse,
            hyperparam_key,
            hyperparam_value,
            num_features,
        )

    print(f"Finished training in {perf_counter() - start:.2f} seconds")

    return result_container
