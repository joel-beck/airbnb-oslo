#%%
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso, LinearRegression, Ridge

simplefilter(action="ignore", category=FutureWarning)

from sklearn_helpers import (
    ModelContainer,
    ResultContainer,
    fit_models,
    get_preprocessor,
)

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

print("Selected Features:")
for feature in selected_features:
    print(feature.split("__")[1])

#%%
# SECTION: Define Models
random_state = 42

linear = ModelContainer(
    LinearRegression(), make_pipeline(preprocessor, selector), "linear", None
)

lasso = ModelContainer(
    Lasso(random_state=random_state),
    make_pipeline(preprocessor, selector),
    "lasso",
    {"lasso__alpha": np.arange(1, 10)},
)

ridge = ModelContainer(
    Ridge(random_state=random_state),
    make_pipeline(preprocessor, selector),
    "ridge",
    {"ridge__alpha": np.arange(100, 500, 50)},
)

models = [linear, lasso, ridge]

# Initialize Results
result_container = ResultContainer()


#%%
# SECTION: Fit Models & Analyze Results
# Since the pipelines include preprocessing and selecting, we fit the model with X and not X_subset
result = fit_models(X, y, models, result_container, n_folds=10)
metrics_df = result.display_results()
metrics_df

#%%
# SUBSECTION: Compare Results with Full Feature Set

# Performance on Validation Set is still bad, but
# maybe slightly better than with full model => Investigate Feature Selection further
pd.read_pickle("full_feature_results.pkl")

#%%
