#%%
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline

simplefilter(action="ignore", category=FutureWarning)

from sklearn_helpers import fit_models, get_preprocessor, get_results

#%%
# SECTION: Select Features
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

# 18 columns in X
X = listings_subset.drop(columns="price")
y = listings_subset["price"]

#%%
# SUBSECTION: Get Selected Features during Model Fitting
# 41 columns in X_processed
preprocessor = get_preprocessor(listings_subset)
X_processed = preprocessor.fit_transform(X, y)

# 10 columns in X_subset
selector = SelectKBest(k=10)
X_subset = selector.fit_transform(X_processed, y)

selected_features = np.array(preprocessor.get_feature_names_out())[
    selector.get_support()
]

for feature in selected_features:
    print(feature)

#%%
# SECTION: Define Models
random_state = 42
linear = LinearRegression()
linear_pipeline = make_pipeline(preprocessor, selector, linear)
linear_grid = None

lasso = Lasso(random_state=random_state)
lasso_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("selector", selector), ("lasso", lasso)]
)
lasso_grid = {"lasso__alpha": range(20, 50, 10)}

models = [linear, lasso]
pipelines = [linear_pipeline, lasso_pipeline]
param_grids = [linear_grid, lasso_grid]

# Initialize Results
model_names = []
grid_key_list = []
grid_value_list = []

r2_train_list = []
r2_val_list = []
mse_train_list = []
mse_val_list = []


#%%
# SECTION: Fit Models
# Since the pipelines include preprocessing and selecting, we fit the model with X and not X_subset
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

metrics_df = get_results(
    model_names,
    r2_train_list,
    r2_val_list,
    mse_train_list,
    mse_val_list,
    grid_key_list,
    grid_value_list,
)
metrics_df

#%%
# SUBSECTION: Compare Results with Full Feature Set

# Performance on Validation Set is still bad, but
# maybe slightly better than with full model => Investigate Feature Selection further
pd.read_pickle("full_feature_results.pkl")

#%%
