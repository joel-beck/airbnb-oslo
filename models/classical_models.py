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

from sklearn_helpers import (
    ModelContainer,
    ResultContainer,
    fit_models,
    get_column_transformer,
)

simplefilter(action="ignore", category=FutureWarning)

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

linear = ModelContainer(LinearRegression(), column_transformer)
lasso = ModelContainer(
    Lasso(random_state=random_state),
    column_transformer,
    {"model__alpha": np.arange(1, 50)},
)

ridge = ModelContainer(
    Ridge(random_state=random_state),
    column_transformer,
    {"model__alpha": np.arange(10, 1000, 10)},
)

random_forest = ModelContainer(
    RandomForestRegressor(random_state=random_state),
    column_transformer,
    {
        "model__max_depth": np.arange(1, 10),
        "model__min_samples_leaf": np.arange(1, 10),
        "model__n_estimators": np.arange(1, 10),
    },
)

gradient_boosting = ModelContainer(
    GradientBoostingRegressor(random_state=random_state),
    column_transformer,
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
    column_transformer,
    {
        "model__learning_rate": np.arange(1, 5),
        "model__n_estimators": np.arange(2, 20, 2),
    },
)

bagging = ModelContainer(
    BaggingRegressor(random_state=random_state),
    column_transformer,
    {
        "model__max_features": np.arange(0.1, 1, 0.1),
        "model__max_samples": np.arange(0.01, 0.1, 0.01),
        "model__n_estimators": np.arange(10, 50, 10),
    },
)

#%%
# SUBSECTION: Collect Instances of Models to iterate over
models = [
    linear,
    lasso,
    ridge,
    random_forest,
    gradient_boosting,
    ada_boost,
    bagging,
]

# Initialize Results with Baseline Model
result_container = ResultContainer(
    model_names=["Mean Prediction"],
    grid_key_list=[None],
    grid_value_list=[None],
    r2_train_list=[baseline_r2],
    r2_val_list=[baseline_r2],
    mse_train_list=[baseline_mse],
    mse_val_list=[baseline_mse],
)

#%%
# SUBSECTION: Fit Models
result = fit_models(X, y, models, result_container, n_folds, n_iter, random_state)
metrics_df = result.display_results()

# save results
metrics_df.to_pickle("full_features_results.pkl")

#%%
# SUBSECTION: Compare Results with Full Feature Set
# import pandas as pd
print("SelectKBest Results:")
pd.read_pickle("k_best_results.pkl")

#%%
print("RFE Results:")
pd.read_pickle("rfe_results.pkl")

#%%
print("Variance Threshold Results:")
pd.read_pickle("vt_results.pkl")

#%%
print("PCA Results:")
pd.read_pickle("pca_results.pkl")

#%%
print("Full Feature Set Results:")
pd.read_pickle("full_features_results.pkl")
