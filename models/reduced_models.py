#%%
from typing import Union
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR

from sklearn_helpers import (
    ModelContainer,
    ResultContainer,
    fit_models,
    get_column_transformer,
    get_feature_selector,
    get_preprocessor,
)

simplefilter(action="ignore", category=FutureWarning)
pd.set_option("precision", 3)

#%%
# BOOKMARK: Hyperparameters
random_state = 42
n_folds = 10
n_iter = 10

# select approximately 50% of all (40) features with each procedure
k = 20
pca_components = 20
rfe_components = 20
vt_threshold = 0.1

#%%
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

# 18 columns in X
X = listings_subset.drop(columns="price")
# TODO: Use Log-Price to Fit the Models
y = listings_subset["price"]

#%%
# SECTION: Explore Selected Features during Model Fitting
column_transformer = get_column_transformer()

#%%
# SUBSECTION: SelectKBest
k_best = get_feature_selector("k_best", k=k)
preprocessor = get_preprocessor(column_transformer, k_best)
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

print("Selected Features from SelectKBest:")
for feature in selected_features:
    print(feature.split("__")[1])

#%%
# SUBSECTION: RFE
rfe = RFE(SVR(kernel="linear"), n_features_to_select=rfe_components, step=0.5)
preprocessor = get_preprocessor(column_transformer, rfe)
preprocessor.fit_transform(X, y)

selected_features = preprocessor.named_steps["feature_selector"].get_feature_names_out(
    encoded_features
)

print("Selected Features from RFE:")
for feature in selected_features:
    print(feature.split("__")[1])

#%%
# SUBSECTION: VarianceThreshold
vt = VarianceThreshold(threshold=vt_threshold)
preprocessor = get_preprocessor(column_transformer, vt)
preprocessor.fit_transform(X, y)

selected_features = preprocessor.named_steps["feature_selector"].get_feature_names_out(
    encoded_features
)

print("Selected Features from VarianceThreshold:")
for feature in selected_features:
    print(feature.split("__")[1])

#%%
# SUBSECTION: PCA
pca = get_feature_selector("pca", pca_components=pca_components)
preprocessor = get_preprocessor(column_transformer, pca)
preprocessor.fit_transform(X, y)

variance_ratios = preprocessor.named_steps["feature_selector"].explained_variance_ratio_

print("Explained Variance Ratios from Principal Components:")
for i, ratio in enumerate(variance_ratios, 1):
    print(f"PC {i}: {np.round(ratio, 3)}")

#%%
# SECTION: Fit Models
# helper function for this file only, depends on global variables
def try_feature_selectors(feature_selector: Union[PCA, SelectKBest]) -> pd.DataFrame:
    column_transformer = get_column_transformer()
    preprocessor = get_preprocessor(column_transformer, feature_selector)

    linear = ModelContainer(LinearRegression(), preprocessor)

    lasso = ModelContainer(
        Lasso(random_state=random_state),
        preprocessor,
        {"model__alpha": np.arange(1, 50)},
    )

    ridge = ModelContainer(
        Ridge(random_state=random_state),
        preprocessor,
        {"model__alpha": np.arange(10, 1000, 10)},
    )

    models = [linear, lasso, ridge]
    result_container = ResultContainer()

    result = fit_models(X, y, models, result_container, n_folds, n_iter, random_state)
    return result.display_results()


#%%
k_best_results = try_feature_selectors(k_best)
k_best_results.to_pickle("k_best_results.pkl")

#%%
rfe_results = try_feature_selectors(rfe)
rfe_results.to_pickle("rfe_results.pkl")

#%%
vt_results = try_feature_selectors(vt)
vt_results.to_pickle("vt_results.pkl")

#%%
pca_results = try_feature_selectors(pca)
pca_results.to_pickle("pca_results.pkl")

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

#%%
