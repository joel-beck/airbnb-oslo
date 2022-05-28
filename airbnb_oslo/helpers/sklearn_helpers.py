from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
    make_column_transformer,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_column_transformer() -> ColumnTransformer:
    """
    Returns ColumnTransformer Object which standardizes all numeric Variables and
    transforms all categorical Variables to Dummy Variables with entries 0 and 1.
    """

    return make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (
            OneHotEncoder(dtype=int, handle_unknown="ignore"),
            make_column_selector(dtype_exclude=np.number),
        ),
    )


def get_feature_selector(
    feature_selector: str,
    pca_components: int | None = None,
    k: int = 10,
) -> PCA | SelectKBest:
    """
    Returns either a PCA or a SelectKBest Object. The number of resulting dimensions
    after application can be specified with input parameters.
    """

    feature_selectors = {"pca": PCA(pca_components), "k_best": SelectKBest(k=k)}
    return feature_selectors[feature_selector]


def get_preprocessor(
    column_transformer: ColumnTransformer, feature_selector: PCA | SelectKBest
) -> Pipeline:
    """
    Creates Pipeline Object that first standardizes all numeric Variables and encodes
    categorical Variables as Dummy Variables and then reduces the Dimensionality of the
    Feature Space.
    """

    return Pipeline(
        [
            ("column_transformer", column_transformer),
            ("feature_selector", feature_selector),
        ]
    )


def update_dict_keys(d: dict, prefix: bool = False, replace: bool = False):
    """
    Returns a new Dictionary with the same Values but different Keys as the Input
    Dictionary. Makes it possible that Hyperparameters can later be specified with their
    original Name instead of the adjusted Name inside of Pipeline Objects.
    """

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
    """
    Collects all Information about a single Regression Model including the applied
    Preprocessing Steps and the Hyperparameters before the Model Fitting Step.
    """

    model: Any
    preprocessor: ColumnTransformer | Pipeline
    param_grid: dict | None = None
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
    """
    Collects all Results of Interest after the Model Fitting Step in a single Object and
    displays them in a Pandas DataFrame
    """

    model_names: list[str] = field(default_factory=list)
    train_mae_list: list[float] = field(default_factory=list)
    val_mae_list: list[float] = field(default_factory=list)
    train_r2_list: list[float] = field(default_factory=list)
    val_r2_list: list[float] = field(default_factory=list)
    train_mse_list: list[float] = field(default_factory=list)
    val_mse_list: list[float] = field(default_factory=list)
    hyperparam_keys: list[str] = field(default_factory=list)
    hyperparam_values: list[float] = field(default_factory=list)
    log_y: list[bool] = field(default_factory=list)
    feature_selector: list[float] = field(default_factory=list)
    num_features: list[float] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)

    def append(
        self,
        train_mae: float,
        val_mae: float,
        train_r2: float,
        val_r2: float,
        train_mse: float,
        val_mse: float,
        hyperparam_key: str | list[str] | None = None,
        hyperparam_value: str | float | list[float] | None = None,
        num_features: int | None = None,
        selected_features: list[str] | None = None,
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
        if selected_features is not None:
            self.selected_features.append(selected_features)

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
                "log_y": self.log_y,
                "feature_selector": self.feature_selector,
                "num_features": self.num_features,
                "selected_features": self.selected_features,
            },
            index=self.model_names,
        )

        return metrics_df.sort_values("mae_val")


def get_models(
    preprocessor: ColumnTransformer | Pipeline,
    models: list[str] | None = None,
    random_state: int | None = None,
    log_y: bool = False,
) -> list[ModelContainer]:
    """
    Returns a List of ModelContainer Objects for all specified Regression Algorithms. If
    the 'models' Parameter is None, all Models are included. This is the only Place
    where the Hyperparameters for each Model as well as their Value Ranges are selected.
    """

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
        {"alpha": np.arange(1, 50)},
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

    hist_gradient_boosting = ModelContainer(
        HistGradientBoostingRegressor(random_state=random_state),
        preprocessor,
        {
            "learning_rate": np.arange(0.01, 0.1, 0.01),
            "max_depth": np.arange(3, 30, 3),
            "max_iter": np.arange(10, 100, 10),
            "max_leaf_nodes": np.arange(5, 50, 5),
            "min_samples_leaf": np.arange(2, 20, 2),
        },
        log_y=log_y,
    )

    model_choice = {
        "linear": linear,
        "lasso": lasso,
        "ridge": ridge,
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting,
        "bagging": bagging,
        "hist_gradient_boosting": hist_gradient_boosting,
    }

    if models is None:
        return list(model_choice.values())

    return [model_choice[model] for model in models]


def get_model_name(model: ModelContainer, log_y: bool):
    """
    Extracts Class Name of a Regression Model contained in a ModelContainer Object
    """

    if log_y:
        return model.model.regressor.__class__.__name__
    else:
        return model.model.__class__.__name__


def get_feature_selector_name(model_container: ModelContainer) -> str | None:
    """
    Extracts Class Name of a Dimensionality Reduction Procedure contained in a
    ModelContainer Object
    """

    try:
        return model_container.preprocessor.named_steps[
            "feature_selector"
        ].__class__.__name__
    # if no feature_selector was used
    except (KeyError, AttributeError):
        return None


def get_selected_features(pipeline: Pipeline) -> list[str]:
    """
    Collects original Variable Names of all Transformed and Selected Features in a List.
    Has to be applied AFTER the Model Pipeline is fitted.
    """
    feature_names = pipeline["preprocessor"][
        "column_transformer"
    ].get_feature_names_out()

    try:
        feature_names = pipeline["preprocessor"][
            "feature_selector"
        ].get_feature_names_out(feature_names)
    # if no feature_selector was used, return names after encoding
    except (KeyError, AttributeError):
        pass

    return [feature.split("__")[1] for feature in feature_names]


def fit_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[ModelContainer],
    result_container: ResultContainer,
    n_folds: int = 5,
    n_iter: int = 10,
    random_state: int | None = None,
    log_y: bool = False,
) -> ResultContainer:
    """
    Fits all specified Models either with Simple Cross Validation, if no Hyperparameters
    are involved, or with Cross Validation using a Randomized Hyperparameter Search to
    find a close to optimal Hyperparameter Combination for each Model. Returns Mean
    Squared Error, Mean Absolute Error and R^2 Value of Training and Validation Set.
    These Metrics are given by the Mean Value across all Folds of the Cross Validation.
    """

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

            selected_features = get_selected_features(scores["estimator"][0])

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

            # number of features after encoding by column_transformer
            num_features = cv.best_estimator_.named_steps["model"].n_features_in_
            selected_features = get_selected_features(cv.best_estimator_)

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
            selected_features,
        )

    print(f"Finished training in {perf_counter() - start:.2f} seconds")

    return result_container


def show_coefficients(log_transform: TransformedTargetRegressor) -> pd.DataFrame:
    """
    Displays Estimated Coefficients of Linear Regression Model in Dataframe.
    """

    encoded_features = log_transform.regressor_.named_steps["pipeline"][
        "column_transformer"
    ].get_feature_names_out()

    selected_features = log_transform.regressor_.named_steps["pipeline"][
        "feature_selector"
    ].get_feature_names_out(encoded_features)

    feature_names = [name.split("__")[1] for name in selected_features]

    coefs = log_transform.regressor_.named_steps["linearregression"].coef_

    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )


def print_metrics(y_true: float, y_hat: float):
    """
    Prints Mean Absolute Error and R^2 Value of Predictions y_hat.
    """

    print(
        f"MAE: {mean_absolute_error(y_true, y_hat):.3f}\n"
        f"R^2: {r2_score(y_true, y_hat):.3f}"
    )
