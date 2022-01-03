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
    get_preprocessor,
)

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

linear = ModelContainer(LinearRegression(), preprocessor, "linear", None)
lasso = ModelContainer(
    Lasso(random_state=random_state),
    preprocessor,
    "lasso",
    {"lasso__alpha": range(20, 50, 10)},
)

ridge = ModelContainer(
    Ridge(random_state=random_state),
    preprocessor,
    "ridge",
    {"ridge__alpha": range(20, 50, 10)},
)

random_forest = ModelContainer(
    RandomForestRegressor(random_state=random_state),
    preprocessor,
    "random_forest",
    {
        "random_forest__n_estimators": [10, 50, 100],
        "random_forest__max_depth": [1, 3, 5],
    },
)

gradient_boosting = ModelContainer(
    GradientBoostingRegressor(random_state=random_state),
    preprocessor,
    "gradient_boosting",
    {"gradient_boosting__max_depth": range(2, 5)},
)

ada_boost = ModelContainer(
    AdaBoostRegressor(random_state=random_state),
    preprocessor,
    "ada_boost",
    {"ada_boost__learning_rate": [1, 2, 3]},
)

bagging = ModelContainer(
    BaggingRegressor(random_state=random_state),
    preprocessor,
    "bagging",
    {"bagging__n_estimators": [10, 20, 50]},
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

#%%
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
# SECTION: Fit Models & Analyze Results
result = fit_models(X, y, models, result_container)
metrics_df = result.display_results()
metrics_df

#%%
# save results
metrics_df.to_pickle("full_feature_results.pkl")

#%%
