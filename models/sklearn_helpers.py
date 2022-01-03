from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [
        "host_gender",
        "host_identity_verified",
        "host_is_superhost",
        "neighbourhood",
        "room_type",
        "shared_bathrooms",
    ]

    numeric_cols = [
        col for col in df.columns if col not in categorical_cols and col != "price"
    ]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_cols),
        (categorical_transformer, categorical_cols),
    )

    return preprocessor


@dataclass
class ModelContainer:
    model: Any
    preprocessor: ColumnTransformer
    pipeline_name: str
    param_grid: Optional[dict] = None

    def __post_init__(self):
        self.pipeline = Pipeline(
            [("preprocessor", self.preprocessor), (self.pipeline_name, self.model)]
        )


@dataclass
class ResultContainer:
    model_names: list[str] = field(default_factory=list)
    grid_key_list: list[str] = field(default_factory=list)
    grid_value_list: list = field(default_factory=list)
    r2_train_list: list[float] = field(default_factory=list)
    r2_val_list: list[float] = field(default_factory=list)
    mse_train_list: list[float] = field(default_factory=list)
    mse_val_list: list[float] = field(default_factory=list)

    def display_results(self) -> pd.DataFrame:
        metrics_df = pd.DataFrame(
            {
                "r2_train": self.r2_train_list,
                "r2_val": self.r2_val_list,
                "mse_train": self.mse_train_list,
                "mse_val": self.mse_val_list,
                "hyperparam_keys": self.grid_key_list,
                "hyperparam_values": self.grid_value_list,
            },
            index=self.model_names,
        )

        return metrics_df.sort_values("r2_val", ascending=False)


def fit_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[ModelContainer],
    result_container: ResultContainer,
) -> ResultContainer:
    start = perf_counter()

    for model in models:
        print("Fitting", model.model.__class__.__name__)
        result_container.model_names.append(model.model.__class__.__name__)

        scoring = ["r2", "neg_mean_squared_error"]

        # if model does not have hyperparameters
        if model.param_grid is None:
            scores = cross_validate(
                model.pipeline, X, y, cv=5, scoring=scoring, return_train_score=True
            )

            r2_train = np.mean(scores["train_r2"])
            r2_val = np.mean(scores["test_r2"])
            mse_train = -np.mean(scores["train_neg_mean_squared_error"])
            mse_val = -np.mean(scores["test_neg_mean_squared_error"])

            hyperparam_key = None
            hyperparam_value = None

        # if model has hyperparameters
        else:
            # if model has exactly one hyperparameter
            if len(model.param_grid) == 1:
                cv = GridSearchCV(
                    estimator=model.pipeline,
                    param_grid=model.param_grid,
                    cv=5,
                    scoring=scoring,
                    refit="neg_mean_squared_error",
                    return_train_score=True,
                )
            # if model has multiple hyperparameters
            else:
                cv = RandomizedSearchCV(
                    estimator=model.pipeline,
                    param_distributions=model.param_grid,
                    n_iter=10,
                    scoring=scoring,
                    refit="neg_mean_squared_error",
                    return_train_score=True,
                )

            cv.fit(X, y)

            hyperparam_key = [key for key in cv.best_params_]
            hyperparam_value = [value for value in cv.best_params_.values()]
            # display scalars in DataFrame instead of one-element lists
            if len(model.param_grid) == 1:
                hyperparam_key = hyperparam_key[0]
                hyperparam_value = hyperparam_value[0]

            best_index = cv.cv_results_["params"].index(cv.best_params_)

            r2_train = cv.cv_results_["mean_train_r2"][best_index]
            r2_val = cv.cv_results_["mean_test_r2"][best_index]
            # for some reason, only negative mean squared error is an available metric
            mse_train = -cv.cv_results_["mean_train_neg_mean_squared_error"][best_index]
            mse_val = -cv.cv_results_["mean_test_neg_mean_squared_error"][best_index]

        result_container.r2_train_list.append(r2_train)
        result_container.r2_val_list.append(r2_val)
        result_container.mse_train_list.append(mse_train)
        result_container.mse_val_list.append(mse_val)
        result_container.grid_key_list.append(hyperparam_key)
        result_container.grid_value_list.append(hyperparam_value)

    print(f"Finished training in {perf_counter() - start:.2f} seconds")

    return result_container
