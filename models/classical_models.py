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
X_train_val = pd.read_pickle("../data-clean/X_train_val.pkl")
y_train_val = pd.read_pickle("../data-clean/y_train_val.pkl")

column_transformer = get_column_transformer()

#%%
# SUBSECTION: Define Models & Hyperparameters
# Baseline Model - Mean Price
# calculated here for all observations, not evaluated on separate test set
def initialize_with_baseline(y_train_val: pd.Series, log_y: bool) -> ResultContainer:
    """
    Creates a new ResultContainer Object and adds Metrics of a Mean-Prediction Baseline Model
    """

    # first log, then average, then transform back to original scale
    mean_price = np.exp(np.mean(np.log(y_train_val))) if log_y else y_train_val.mean()
    baseline_pred = np.full(shape=y_train_val.shape, fill_value=mean_price)
    baseline_r2 = r2_score(y_true=y_train_val, y_pred=baseline_pred)
    baseline_mse = mean_squared_error(y_true=y_train_val, y_pred=baseline_pred)
    baseline_mae = mean_absolute_error(y_true=y_train_val, y_pred=baseline_pred)

    result_container = ResultContainer(
        model_names=["Mean Prediction"],
        train_r2_list=[baseline_r2],
        val_r2_list=[baseline_r2],
        train_mse_list=[baseline_mse],
        val_mse_list=[baseline_mse],
        train_mae_list=[baseline_mae],
        val_mae_list=[baseline_mae],
        hyperparam_keys=[None],
        hyperparam_values=[None],
        num_features=[None],
        feature_selector=[None],
        log_y=[log_y],
    )

    return result_container


#%%
# SUBSECTION: Fit Models
full_features_results = []
for log_y in [True, False]:
    result_container = initialize_with_baseline(y_train_val, log_y=log_y)
    models = get_models(column_transformer, random_state=random_state, log_y=log_y)
    result = fit_models(
        X_train_val,
        y_train_val,
        models,
        result_container,
        n_folds,
        n_iter,
        random_state,
        log_y=log_y,
    )
    full_features_results.append(result.display_df())

pd.concat(full_features_results).to_pickle("../results-pickle/full_features_results.pkl")

#%%
