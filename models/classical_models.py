#%%
# model fitting at the end produces
# FutureWarning: Arrays of bytes/strings is being converted to decimal numbers if
# dtype='numeric'.
# This behavior is deprecated in 0.24 and will be removed in 1.1 (renaming of 0.26).
# Please convert your data to numeric values explicitly instead.
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn_helpers import (
    ResultContainer,
    fit_models,
    get_column_transformer,
    get_models,
)

simplefilter(action="ignore", category=FutureWarning)
pd.set_option("precision", 3)

#%%
# BOOKMARK: Hyperparameters
random_state = 42
n_folds = 10
n_iter = 10

#%%
# SUBSECTION: Transform Categorical Columns to Dummies and Standardize Numeric Columns
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

X = listings_subset.drop(columns="price")
y = listings_subset["price"]

column_transformer = get_column_transformer()

#%%
# SUBSECTION: Define Models & Hyperparameters
# Baseline Model - Mean Price
# calculated here for all observations, not evaluated on separate test set
mean_price = y.mean()
baseline_pred = np.full(shape=y.shape, fill_value=mean_price)
baseline_r2 = r2_score(y_true=y, y_pred=baseline_pred)
baseline_mse = mean_squared_error(y_true=y, y_pred=baseline_pred)
baseline_mae = mean_absolute_error(y_true=y, y_pred=baseline_pred)

#%%
# SUBSECTION: Collect Instances of Models to iterate over
models = get_models(column_transformer, random_state=random_state)

# Initialize Results with Baseline Model
result_container = ResultContainer(
    model_names=["Mean Prediction"],
    train_r2_list=[baseline_r2],
    val_r2_list=[baseline_r2],
    train_mse_list=[baseline_mse],
    val_mse_list=[baseline_mse],
    train_mae_list=[baseline_mae],
    val_mae_list=[baseline_mae],
    grid_key_list=[None],
    grid_value_list=[None],
    num_features=[None],
    feature_selector=[None],
)

#%%
# SUBSECTION: Fit Models
result = fit_models(X, y, models, result_container, n_folds, n_iter, random_state)
metrics_df = result.display_results()

# save results
metrics_df.to_pickle("full_features_results.pkl")
