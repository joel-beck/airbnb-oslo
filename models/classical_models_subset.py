#%%
import itertools
from typing import Union
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    VarianceThreshold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

from sklearn_helpers import (
    ResultContainer,
    fit_models,
    get_column_transformer,
    get_feature_selector,
    get_models,
    get_preprocessor,
)

simplefilter(action="ignore", category=FutureWarning)
pd.set_option("display.precision", 3)

#%%
# BOOKMARK: Hyperparameters
random_state = 42
n_folds = 10
n_iter = 10

# select approximately 50% of all (40) features with each procedure
k_list = [10, 20, 30]
pca_components_list = [10, 20, 30]
rfe_components_list = [10, 20, 30]
sfm_threshold_list = [
    str(num) + "*mean" if num != 1 else "mean" for num in [0.5, 1, 1.5]
]
sfs_components_list = [10, 20, 30]  # currently not used
vt_threshold_list = [0.1, 0.2, 0.3]
log_y_list = [True, False]

#%%
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

# 18 columns before Encoding
X_train_val = pd.read_pickle("../data-clean/X_train_val.pkl")
y_train_val = pd.read_pickle("../data-clean/y_train_val.pkl")

#%%
# SECTION: Explore Selected Features during Model Fitting
column_transformer = get_column_transformer()

#%%
# SUBSECTION: SelectKBest
k_best = get_feature_selector("k_best", k=10)
preprocessor = get_preprocessor(column_transformer, k_best)
preprocessor.fit_transform(X_train_val, y_train_val)

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
rfe = RFE(SVR(kernel="linear"), n_features_to_select=10, step=0.5)
preprocessor = get_preprocessor(column_transformer, rfe)
preprocessor.fit_transform(X_train_val, y_train_val)

selected_features = preprocessor.named_steps["feature_selector"].get_feature_names_out(
    encoded_features
)

print("Selected Features from RFE:")
for feature in selected_features:
    print(feature.split("__")[1])

#%%
# SUBSECTION: SelectFromModel
sfm = SelectFromModel(SVR(kernel="linear"), threshold="1.5*mean")
preprocessor = get_preprocessor(column_transformer, sfm)
preprocessor.fit_transform(X_train_val, y_train_val)

selected_features = preprocessor.named_steps["feature_selector"].get_feature_names_out(
    encoded_features
)

print("Selected Features from SelectFromModel:")
for feature in selected_features:
    print(feature.split("__")[1])

#%%
# SUBSECTION: SequentialFeatureSelector
sfs = SequentialFeatureSelector(estimator=LogisticRegression(), n_features_to_select=10)
preprocessor = get_preprocessor(column_transformer, sfm)
preprocessor.fit_transform(X_train_val, y_train_val)

selected_features = preprocessor.named_steps["feature_selector"].get_feature_names_out(
    encoded_features
)

print("Selected Features from SequentialFeatureSelector:")
for feature in selected_features:
    print(feature.split("__")[1])

#%%
# SUBSECTION: VarianceThreshold
vt = VarianceThreshold(threshold=0.3)
preprocessor = get_preprocessor(column_transformer, vt)
preprocessor.fit_transform(X_train_val, y_train_val)

selected_features = preprocessor.named_steps["feature_selector"].get_feature_names_out(
    encoded_features
)

print("Selected Features from VarianceThreshold:")
for feature in selected_features:
    print(feature.split("__")[1])

#%%
# SUBSECTION: PCA
pca = get_feature_selector("pca", pca_components=10)
preprocessor = get_preprocessor(column_transformer, pca)
preprocessor.fit_transform(X_train_val, y_train_val)

variance_ratios = preprocessor.named_steps["feature_selector"].explained_variance_ratio_

print("Explained Variance Ratios from Principal Components:")
for i, ratio in enumerate(variance_ratios, 1):
    print(f"PC {i}: {np.round(ratio, 3)}")

#%%
# SECTION: Fit Models
# helper function for this file only, depends on global variables
def try_feature_selectors(
    feature_selector: Union[PCA, SelectKBest], log_y: bool = False
) -> pd.DataFrame:
    """
    Reduces Dimensionality of Feature Space with a given Feature Selector, fits all Models including Hyperparameter Tuning and returns a Pandas DataFrame with the Results.
    This function is used to concatenate the Results for all Feature Selectors to compare their effectiveness.
    """

    column_transformer = get_column_transformer()
    preprocessor = get_preprocessor(column_transformer, feature_selector)

    models = get_models(preprocessor, random_state=random_state, log_y=log_y)
    result_container = ResultContainer()

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
    return result.display_df()


#%%
k_best_results = []
for (k, log_y) in itertools.product(k_list, log_y_list):
    k_best = get_feature_selector("k_best", k=k)
    k_best_results.append(try_feature_selectors(k_best, log_y=log_y))

pd.concat(k_best_results).to_pickle("../results-pickle/k_best_results.pkl")

#%%
rfe_results = []
for (rfe_components, log_y) in itertools.product(rfe_components_list, log_y_list):
    rfe = RFE(SVR(kernel="linear"), n_features_to_select=rfe_components, step=0.5)
    rfe_results.append(try_feature_selectors(rfe, log_y=log_y))

pd.concat(rfe_results).to_pickle("../results-pickle/rfe_results.pkl")

#%%
sfm_results = []
for (sfm_threshold, log_y) in itertools.product(sfm_threshold_list, log_y_list):
    sfm = SelectFromModel(SVR(kernel="linear"), threshold=sfm_threshold)
    sfm_results.append(try_feature_selectors(sfm, log_y=log_y))

pd.concat(sfm_results).to_pickle("../results-pickle/sfm_results.pkl")

#%%
vt_results = []
for (vt_threshold, log_y) in itertools.product(vt_threshold_list, log_y_list):
    vt = VarianceThreshold(threshold=vt_threshold)
    vt_results.append(try_feature_selectors(vt, log_y=log_y))

pd.concat(vt_results).to_pickle("../results-pickle/vt_results.pkl")

#%%
pca_results = []
for (pca_components, log_y) in itertools.product(pca_components_list, log_y_list):
    pca = get_feature_selector("pca", pca_components=pca_components)
    pca_results.append(try_feature_selectors(pca, log_y=log_y))

pd.concat(pca_results).to_pickle("../results-pickle/pca_results.pkl")

#%%
