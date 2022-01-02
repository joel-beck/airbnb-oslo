#%%
from time import perf_counter

# model fitting at the end produces
# FutureWarning: Arrays of bytes/strings is being converted to decimal numbers if
# dtype='numeric'.
# This behavior is deprecated in 0.24 and will be removed in 1.1 (renaming of 0.26).
# Please convert your data to numeric values explicitly instead.
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

simplefilter(action="ignore", category=FutureWarning)

#%%
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

X = listings_subset.drop(columns="price")
y = listings_subset["price"]

#%%
# SECTION: Transform Categorical Columns to Dummies and Standardize Numeric Columns
# NOTE: This section of the code is currently duplicated with the pytorch_preprocessing file => maybe find solution to remove that redundancy
categorical_cols = [
    "host_gender",
    "host_identity_verified",
    "host_is_superhost",
    "neighbourhood",
    "room_type",
    "shared_bathrooms",
]

numeric_cols = [
    col
    for col in listings_subset.columns
    if col not in categorical_cols and col != "price"
]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_cols),
    (categorical_transformer, categorical_cols),
)

random_state = 123

#%%
# SECTION: Model Fit & Hyperparameter Search
# SUBSECTION: Define Models & Hyperparameters
# Baseline Model - Mean Price
# calculated here for all observations, not evaluated on separate test set
mean_price = y.mean()
baseline_pred = np.full(shape=y.shape, fill_value=mean_price)
baseline_r2 = r2_score(y_true=y, y_pred=baseline_pred)
baseline_mse = mean_squared_error(y_true=y, y_pred=baseline_pred)

linear = LinearRegression()
linear_pipeline = make_pipeline(preprocessor, linear)
linear_grid = None

lasso = Lasso(random_state=random_state)
lasso_pipeline = Pipeline([("preprocessor", preprocessor), ("lasso", lasso)])
lasso_grid = {"lasso__alpha": range(10, 60, 10)}

ridge = Ridge(random_state=random_state)
ridge_pipeline = Pipeline([("preprocessor", preprocessor), ("ridge", ridge)])
ridge_grid = {"ridge__alpha": range(10, 60, 10)}

# TODO: Search over multiple hyperparameters and try RandomizedSearch
random_forest = RandomForestRegressor(random_state=random_state)
random_forest_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("random_forest", random_forest)]
)
random_forest_grid = {
    "random_forest__n_estimators": [10, 50, 100],
    "random_forest__max_depth": [1, 3, 5],
}

gradient_boosting = GradientBoostingRegressor(random_state=random_state)
gradient_boosting_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("gradient_boosting", gradient_boosting)]
)
gradient_boosting_grid = {"gradient_boosting__max_depth": range(1, 6)}

ada_boost = AdaBoostRegressor(random_state=random_state)
ada_boost_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("ada_boost", ada_boost)]
)
ada_boost_grid = {"ada_boost__learning_rate": [0.5, 1, 2, 3, 5]}

bagging = BaggingRegressor(random_state=random_state)
bagging_pipeline = Pipeline([("preprocessor", preprocessor), ("bagging", bagging)])
bagging_grid = {"bagging__n_estimators": [5, 10, 20, 50]}

#%%
# SUBSECTION: Collect Models
models = [
    linear,
    lasso,
    ridge,
    random_forest,
    gradient_boosting,
    ada_boost,
    bagging,
]

pipelines = [
    linear_pipeline,
    lasso_pipeline,
    ridge_pipeline,
    random_forest_pipeline,
    gradient_boosting_pipeline,
    ada_boost_pipeline,
    bagging_pipeline,
]

param_grids = [
    linear_grid,
    lasso_grid,
    ridge_grid,
    random_forest_grid,
    gradient_boosting_grid,
    ada_boost_grid,
    bagging_grid,
]

#%%
# SUBSECTION: Initialize Results with Baseline Model
model_names = ["Mean Prediction"]
grid_key_list = [None]
grid_value_list = [None]

r2_train_list = [baseline_r2]
r2_test_list = [baseline_r2]
mse_train_list = [baseline_mse]
mse_test_list = [baseline_mse]

#%%
# SUBSECTION: Fit Models
start = perf_counter()

for (model, pipeline, param_grid) in zip(models, pipelines, param_grids):
    print("Fitting", model.__class__.__name__)
    model_names.append(model.__class__.__name__)

    scoring = ["r2", "neg_mean_squared_error"]

    # if model has hyperparameters
    if param_grid is not None:
        cv = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            refit="neg_mean_squared_error",
            return_train_score=True,
        )
        cv.fit(X, y)

        best_index = cv.cv_results_["params"].index(cv.best_params_)

        r2_train = cv.cv_results_["mean_train_r2"][best_index]
        r2_test = cv.cv_results_["mean_test_r2"][best_index]
        # for some reason, only negative mean squared error is an available metric
        mse_train = -cv.cv_results_["mean_train_neg_mean_squared_error"][best_index]
        mse_test = -cv.cv_results_["mean_test_neg_mean_squared_error"][best_index]

        hyperparam_key = [key for key in cv.best_params_]
        hyperparam_value = [value for value in cv.best_params_.values()]

        if len(hyperparam_key) == 1:
            hyperparam_key = hyperparam_key[0]
            hyperparam_value = hyperparam_value[0]

    # if model does not have hyperparameters
    else:
        scores = cross_validate(
            pipeline, X, y, cv=5, scoring=scoring, return_train_score=True
        )

        r2_train = np.mean(scores["train_r2"])
        r2_test = np.mean(scores["test_r2"])
        mse_train = -np.mean(scores["train_neg_mean_squared_error"])
        mse_test = -np.mean(scores["test_neg_mean_squared_error"])

        hyperparam_key = None
        hyperparam_value = None

    r2_train_list.append(r2_train)
    r2_test_list.append(r2_test)
    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    grid_key_list.append(hyperparam_key)
    grid_value_list.append(hyperparam_value)

print(f"Finished training in {perf_counter() - start:.2f} seconds")


#%%
# SUBSECTION: Analyze Results
metrics_df = pd.DataFrame(
    {
        "r2_train": r2_train_list,
        "r2_test": r2_test_list,
        "mse_train": mse_train_list,
        "mse_test": mse_test_list,
        "hyperparam_key": grid_key_list,
        "hyperparam_value": grid_value_list,
    },
    index=model_names,
)

# NOTE: Performances on validation set terrible right now => probably design mistake
metrics_df.sort_values("r2_test", ascending=False)

#%%
# save results
metrics_df.to_pickle("classical_models_results.pkl")

#%%
