from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import (
    ColumnTransformer,
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


@dataclass
class ModelContainer:
    model: Any
    preprocessor: Union[ColumnTransformer, Pipeline]
    param_grid: Optional[dict] = None

    def __post_init__(self):
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
    grid_key_list: list[str] = field(default_factory=list)
    grid_value_list: list = field(default_factory=list)
    num_features: list[float] = field(default_factory=list)
    feature_selector: list[float] = field(default_factory=list)

    def update_metrics(
        self,
        train_mae: float,
        val_mae: float,
        train_r2: float,
        val_r2: float,
        train_loss: float,
        val_loss: float,
    ):
        self.train_mae_list.append(train_mae)
        self.val_mae_list.append(val_mae)
        self.train_r2_list.append(train_r2)
        self.val_r2_list.append(val_r2)
        self.train_mse_list.append(train_loss)
        self.val_mse_list.append(val_loss)

    def display_df(self) -> pd.DataFrame:
        metrics_df = pd.DataFrame(
            {
                "mae_train": self.train_mae_list,
                "mae_val": self.val_mae_list,
                "r2_train": self.train_r2_list,
                "r2_val": self.val_r2_list,
                "mse_train": self.train_mse_list,
                "mse_val": self.val_mse_list,
                "hyperparams": self.grid_key_list,
                "hyperparam_values": self.grid_value_list,
                "num_features": self.num_features,
                "feature_selector": self.feature_selector,
            },
            index=self.model_names,
        )

        return metrics_df.sort_values("mae_val")


def get_feature_selector_name(model_container: ModelContainer) -> Optional[str]:
    try:
        return model_container.preprocessor.named_steps[
            "feature_selector"
        ].__class__.__name__
    # if no feature_selector was used
    except (KeyError, AttributeError):
        return None


def get_models(
    preprocessor: Union[ColumnTransformer, Pipeline],
    models: Optional[list[str]] = None,
    random_state: Optional[int] = None,
) -> list[ModelContainer]:

    linear = ModelContainer(LinearRegression(), preprocessor)
    lasso = ModelContainer(
        Lasso(random_state=random_state),
        preprocessor,
        {"model__alpha": np.arange(1, 50)},
    )

    ridge = ModelContainer(
        Ridge(random_state=random_state),
        preprocessor,
        {"model__alpha": np.arange(10, 1000, 10)},
    )

    random_forest = ModelContainer(
        RandomForestRegressor(random_state=random_state),
        preprocessor,
        {
            "model__max_depth": np.arange(1, 10),
            "model__min_samples_leaf": np.arange(1, 10),
            "model__n_estimators": np.arange(1, 10),
        },
    )

    gradient_boosting = ModelContainer(
        GradientBoostingRegressor(random_state=random_state),
        preprocessor,
        {
            "model__learning_rate": np.arange(0.1, 1, 0.1),
            "model__max_depth": np.arange(1, 10),
            "model__min_samples_leaf": np.arange(1, 10),
            "model__n_estimators": np.arange(1, 10),
            "model__subsample": np.arange(0.01, 0.2, 0.02),
        },
    )

    ada_boost = ModelContainer(
        AdaBoostRegressor(random_state=random_state),
        preprocessor,
        {
            "model__learning_rate": np.arange(1, 5),
            "model__n_estimators": np.arange(2, 20, 2),
        },
    )

    bagging = ModelContainer(
        BaggingRegressor(random_state=random_state),
        preprocessor,
        {
            "model__max_features": np.arange(0.1, 1, 0.1),
            "model__max_samples": np.arange(0.01, 0.1, 0.01),
            "model__n_estimators": np.arange(10, 50, 10),
        },
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
        return model_choice.values()

    return [model_choice[model] for model in models]


def fit_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[ModelContainer],
    result_container: ResultContainer,
    n_folds: int = 5,
    n_iter: int = 10,
    random_state: Optional[int] = None,
) -> ResultContainer:
    start = perf_counter()

    for model in models:
        print("Fitting", model.model.__class__.__name__)
        result_container.model_names.append(model.model.__class__.__name__)
        result_container.feature_selector.append(get_feature_selector_name(model))

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

            mae_train = -np.mean(scores["train_neg_mean_absolute_error"])
            mae_val = -np.mean(scores["test_neg_mean_absolute_error"])
            r2_train = np.mean(scores["train_r2"])
            r2_val = np.mean(scores["test_r2"])
            mse_train = -np.mean(scores["train_neg_mean_squared_error"])
            mse_val = -np.mean(scores["test_neg_mean_squared_error"])

            hyperparam_key = None
            hyperparam_value = None
            num_features = scores["estimator"][0]["model"].n_features_in_

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
            mae_train = -cv.cv_results_["mean_train_neg_mean_absolute_error"][
                best_index
            ]
            mae_val = -cv.cv_results_["mean_test_neg_mean_absolute_error"][best_index]
            r2_train = cv.cv_results_["mean_train_r2"][best_index]
            r2_val = cv.cv_results_["mean_test_r2"][best_index]
            mse_train = -cv.cv_results_["mean_train_neg_mean_squared_error"][best_index]
            mse_val = -cv.cv_results_["mean_test_neg_mean_squared_error"][best_index]

            hyperparam_key = [key.split("__")[1] for key in cv.best_params_]
            hyperparam_value = [value for value in cv.best_params_.values()]
            num_features = cv.best_estimator_.named_steps["model"].n_features_in_

            # display scalars in DataFrame instead of one-element lists
            if len(model.param_grid) == 1:
                hyperparam_key = hyperparam_key[0]
                hyperparam_value = hyperparam_value[0]

        result_container.train_mae_list.append(mae_train)
        result_container.val_mae_list.append(mae_val)
        result_container.train_r2_list.append(r2_train)
        result_container.val_r2_list.append(r2_val)
        result_container.train_mse_list.append(mse_train)
        result_container.val_mse_list.append(mse_val)
        result_container.grid_key_list.append(hyperparam_key)
        result_container.grid_value_list.append(hyperparam_value)
        result_container.num_features.append(num_features)

    print(f"Finished training in {perf_counter() - start:.2f} seconds")

    return result_container
