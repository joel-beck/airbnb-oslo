#%%
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

simplefilter(action="ignore", category=FutureWarning)

from sklearn_helpers import (
    ModelContainer,
    ResultContainer,
    fit_models,
    get_column_transformer,
    get_feature_selector,
    get_preprocessor,
)

#%%
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

# 18 columns in X
X = listings_subset.drop(columns="price")
y = listings_subset["price"]

# BOOKMARK: Hyperparameters
random_state = 42
n_folds = 10
k = 20
pca_components = 20

#%%
# SECTION: Part 1: SelectKBest
# SUBSECTION: Get Selected Features during Model Fitting
column_transformer = get_column_transformer()
feature_selector = get_feature_selector("k_best", k=k)
preprocessor = get_preprocessor(column_transformer, feature_selector)

preprocessor.fit_transform(X, y)

# 40 columns after One-Hot Encoding
encoded_features = preprocessor.named_steps[
    "column_transformer"
].get_feature_names_out()
len(encoded_features)

#%%
selected_features = preprocessor.named_steps["feature_selector"].get_feature_names_out(
    encoded_features
)

print("Selected Features:")
for feature in selected_features:
    print(feature.split("__")[1])

#%%
# SUBSECTION: Define Models
linear = ModelContainer(LinearRegression(), preprocessor)

lasso = ModelContainer(
    Lasso(random_state=random_state),
    preprocessor,
    {"model__alpha": np.arange(1, 10)},
)

ridge = ModelContainer(
    Ridge(random_state=random_state),
    preprocessor,
    {"model__alpha": np.arange(50, 500, 50)},
)

models = [linear, lasso, ridge]

# Initialize Results
result_container = ResultContainer()

#%%
# SUBSECTION: Fit Models & Analyze Results
result = fit_models(X, y, models, result_container, n_folds=n_folds)
metrics_df = result.display_results()

# save results
metrics_df.to_pickle("k_best_results.pkl")

#%%
# SECTION: Part 2: PCA
feature_selector = get_feature_selector("pca", pca_components=pca_components)
preprocessor = get_preprocessor(column_transformer, feature_selector)

linear = ModelContainer(LinearRegression(), preprocessor)

lasso = ModelContainer(
    Lasso(random_state=random_state),
    preprocessor,
    {"model__alpha": np.arange(1, 10)},
)

ridge = ModelContainer(
    Ridge(random_state=random_state),
    preprocessor,
    {"model__alpha": np.arange(50, 500, 50)},
)

models = [linear, lasso, ridge]
result_container = ResultContainer()

result = fit_models(X, y, models, result_container, n_folds=n_folds)
metrics_df = result.display_results()

metrics_df.to_pickle("pca_results.pkl")

#%%
# SUBSECTION: Compare Results with Full Feature Set
# import pandas as pd
pd.read_pickle("k_best_results.pkl")

#%%
pd.read_pickle("pca_results.pkl")

#%%
pd.read_pickle("full_features_results.pkl")

#%%
