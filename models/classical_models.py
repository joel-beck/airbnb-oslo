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
model_names = []
grid_key_list = []
grid_value_list = []

r2_train_list = []
r2_test_list = []
mse_train_list = []
mse_test_list = []

#%%
# SUBSECTION: Define Models & Hyperparameters
linear = LinearRegression()
linear_pipeline = make_pipeline(preprocessor, linear)
linear_key = None
linear_values = None

lasso = Lasso(random_state=random_state)
lasso_pipeline = Pipeline([("preprocessor", preprocessor), ("lasso", lasso)])
lasso_key = "lasso__alpha"
lasso_values = [1, 5, 10, 20, 50]

ridge = Ridge(random_state=random_state)
ridge_pipeline = Pipeline([("preprocessor", preprocessor), ("ridge", ridge)])
ridge_key = "ridge__alpha"
ridge_values = [1, 5, 10, 20, 50]

# TODO: Generalize Design to multiple hyperparameters in GridSearch
# TODO: Maybe use RandomizedSearch
random_forest = RandomForestRegressor(random_state=random_state)
random_forest_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("random_forest", random_forest)]
)
random_forest_key = "random_forest__max_depth"
random_forest_values = [3, 5, 10]

# TODO: Add additional models
# ada_boost = AdaBoostRegressor(random_state=random_state)
# bagging = BaggingRegressor(random_state=random_state)
# gradient_boosting = GradientBoostingRegressor(random_state=random_state)

models = [linear, lasso, ridge, random_forest]
pipelines = [linear_pipeline, lasso_pipeline, ridge_pipeline, random_forest_pipeline]
grid_keys = [linear_key, lasso_key, ridge_key, random_forest_key]
grid_values = [linear_values, lasso_values, ridge_values, random_forest_values]

#%%
# SUBSECTION: Fit Models
start = perf_counter()

for (model, pipeline, grid_key, grid_value) in zip(
    models, pipelines, grid_keys, grid_values
):
    print("Fitting", model.__class__.__name__)
    model_names.append(model.__class__.__name__)

    scoring = ["r2", "neg_mean_squared_error"]

    # if model has hyperparameters
    if grid_value is not None:
        cv = GridSearchCV(
            estimator=pipeline,
            param_grid={grid_key: grid_value},
            cv=10,
            scoring=scoring,
            refit="neg_mean_squared_error",
            return_train_score=True,
        )
        cv.fit(X, y)

        best_value = cv.best_params_[grid_key]
        best_index = grid_value.index(best_value)

        r2_train = cv.cv_results_["mean_train_r2"][best_index]
        r2_test = cv.cv_results_["mean_test_r2"][best_index]
        # for some reason, only negative mean squared error is an available metric
        mse_train = -cv.cv_results_["mean_train_neg_mean_squared_error"][best_index]
        mse_test = -cv.cv_results_["mean_test_neg_mean_squared_error"][best_index]

    # if model does not have hyperparameters
    else:
        scores = cross_validate(
            pipeline, X, y, cv=10, scoring=scoring, return_train_score=True
        )
        best_value = None

        r2_train = np.mean(scores["train_r2"])
        r2_test = np.mean(scores["test_r2"])
        mse_train = -np.mean(scores["train_neg_mean_squared_error"])
        mse_test = -np.mean(scores["test_neg_mean_squared_error"])

    grid_key_list.append(grid_key)
    grid_value_list.append(best_value)

    r2_train_list.append(r2_train)
    r2_test_list.append(r2_test)
    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)

print(f"Finished training in {perf_counter() - start: .2f} seconds")


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

# NOTE: Performances terrible right now => probably design mistake
metrics_df.sort_values("r2_test", ascending=False)

#%%
# save results
metrics_df.to_pickle("classical_models_results.pkl")
