from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_preprocessor(df):
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


def fit_models(
    X,
    y,
    models,
    pipelines,
    param_grids,
    model_names,
    grid_key_list,
    grid_value_list,
    r2_train_list,
    r2_val_list,
    mse_train_list,
    mse_val_list,
):
    start = perf_counter()

    for (model, pipeline, param_grid) in zip(models, pipelines, param_grids):
        print("Fitting", model.__class__.__name__)
        model_names.append(model.__class__.__name__)

        scoring = ["r2", "neg_mean_squared_error"]

        # if model does not have hyperparameters
        if param_grid is None:
            scores = cross_validate(
                pipeline, X, y, cv=5, scoring=scoring, return_train_score=True
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
            if len(param_grid) == 1:
                cv = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=5,
                    scoring=scoring,
                    refit="neg_mean_squared_error",
                    return_train_score=True,
                )
            # if model has multiple hyperparameters
            else:
                cv = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    n_iter=10,
                    scoring=scoring,
                    refit="neg_mean_squared_error",
                    return_train_score=True,
                )

            cv.fit(X, y)

            hyperparam_key = [key for key in cv.best_params_]
            hyperparam_value = [value for value in cv.best_params_.values()]
            # display scalars in DataFrame instead of one-element lists
            if len(param_grid) == 1:
                hyperparam_key = hyperparam_key[0]
                hyperparam_value = hyperparam_value[0]

            best_index = cv.cv_results_["params"].index(cv.best_params_)

            r2_train = cv.cv_results_["mean_train_r2"][best_index]
            r2_val = cv.cv_results_["mean_test_r2"][best_index]
            # for some reason, only negative mean squared error is an available metric
            mse_train = -cv.cv_results_["mean_train_neg_mean_squared_error"][best_index]
            mse_val = -cv.cv_results_["mean_test_neg_mean_squared_error"][best_index]

        r2_train_list.append(r2_train)
        r2_val_list.append(r2_val)
        mse_train_list.append(mse_train)
        mse_val_list.append(mse_val)
        grid_key_list.append(hyperparam_key)
        grid_value_list.append(hyperparam_value)

    print(f"Finished training in {perf_counter() - start:.2f} seconds")

    return (
        r2_train_list,
        r2_val_list,
        mse_train_list,
        mse_val_list,
        grid_key_list,
        grid_value_list,
    )


def get_results(
    model_names,
    r2_train_list,
    r2_val_list,
    mse_train_list,
    mse_val_list,
    grid_key_list,
    grid_value_list,
):
    metrics_df = pd.DataFrame(
        {
            "r2_train": r2_train_list,
            "r2_val": r2_val_list,
            "mse_train": mse_train_list,
            "mse_val": mse_val_list,
            "hyperparam_keys": grid_key_list,
            "hyperparam_values": grid_value_list,
        },
        index=model_names,
    )

    return metrics_df.sort_values("r2_val", ascending=False)
