#%%
# model fitting at the end produces
# FutureWarning: Arrays of bytes/strings is being converted to decimal numbers if
# dtype='numeric'.
# This behavior is deprecated in 0.24 and will be removed in 1.1 (renaming of 0.26).
# Please convert your data to numeric values explicitly instead.
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn_helpers import fit_models, get_preprocessor, get_results

simplefilter(action="ignore", category=FutureWarning)

#%%
# SUBSECTION: Transform Categorical Columns to Dummies and Standardize Numeric Columns
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")
preprocessor = get_preprocessor(listings_subset)

X = listings_subset.drop(columns="price")
y = listings_subset["price"]

#%%
# SUBSECTION: Define Models & Hyperparameters
random_state = 123

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
lasso_grid = {"lasso__alpha": range(20, 50, 10)}

ridge = Ridge(random_state=random_state)
ridge_pipeline = Pipeline([("preprocessor", preprocessor), ("ridge", ridge)])
ridge_grid = {"ridge__alpha": range(20, 50, 10)}

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
gradient_boosting_grid = {"gradient_boosting__max_depth": range(2, 5)}

ada_boost = AdaBoostRegressor(random_state=random_state)
ada_boost_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("ada_boost", ada_boost)]
)
ada_boost_grid = {"ada_boost__learning_rate": [1, 2, 3]}

bagging = BaggingRegressor(random_state=random_state)
bagging_pipeline = Pipeline([("preprocessor", preprocessor), ("bagging", bagging)])
bagging_grid = {"bagging__n_estimators": [10, 20, 50]}

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
# Initialize Results with Baseline Model
model_names = ["Mean Prediction"]
grid_key_list = [None]
grid_value_list = [None]

r2_train_list = [baseline_r2]
r2_val_list = [baseline_r2]
mse_train_list = [baseline_mse]
mse_val_list = [baseline_mse]

#%%
# SUBSECTION: Fit Models
# Sometimes formatting with black looks ugly ^^
(
    r2_train_list,
    r2_val_list,
    mse_train_list,
    mse_val_list,
    grid_key_list,
    grid_value_list,
) = fit_models(
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
)


#%%
# SUBSECTION: Analyze and Save Results
# NOTE: Performances on validation set terrible right now => probably design mistake
metrics_df = get_results(
    model_names,
    r2_train_list,
    r2_val_list,
    mse_train_list,
    mse_val_list,
    grid_key_list,
    grid_value_list,
)
print(metrics_df)

#%%
# save results
metrics_df.to_pickle("full_feature_results.pkl")

#%%

metrics_df = pd.read_pickle("models/full_feature_results.pkl")
