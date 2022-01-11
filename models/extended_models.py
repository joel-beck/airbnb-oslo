#%%
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

from sklearn_helpers import (
    ResultContainer,
    fit_models,
    get_column_transformer,
    get_models,
    get_preprocessor,
    show_coefficients,
)

simplefilter(action="ignore", category=FutureWarning)
pd.set_option("precision", 3)


#%%
# NOTE: For Experimentation we train model on the entire data set without splitting in training and test set
listings_extended = pd.read_pickle("../data-clean/listings_extended.pkl")
X = listings_extended.drop(columns="price")
y = listings_extended["price"]


#%%
# BOOKMARK: Hyperparameters
random_state = 42
n_folds = 10
log_y = True

# 131 total encoded features in listings_extended
# using all 131 features leads to error
num_features_list = [10, 20, 50, 100]


#%%
column_transformer = get_column_transformer()


#%%
# SUBSECTION: Analyze Performance for different values of num_features
result_list = []
for num_features in num_features_list:
    rfe = RFE(SVR(kernel="linear"), n_features_to_select=num_features, step=0.5)
    preprocessor = get_preprocessor(column_transformer, rfe)
    models = get_models(
        preprocessor, models=["linear"], random_state=random_state, log_y=log_y
    )
    result_container = ResultContainer()

    result = fit_models(
        X,
        y,
        models,
        result_container,
        n_folds,
        random_state=random_state,
        log_y=log_y,
    )
    result_list.append(result.display_df())

collected_results = pd.concat(result_list)


#%%
collected_results.sort_values("mae_val")


#%%
# SUBSECTION: Analyze Coefficients for different values of num_features
num_features = 20
rfe = RFE(SVR(kernel="linear"), n_features_to_select=num_features, step=0.5)
preprocessor = get_preprocessor(column_transformer, rfe)
model = LinearRegression()

pipeline = make_pipeline(preprocessor, model)
log_transform = TransformedTargetRegressor(pipeline, func=np.log, inverse_func=np.exp)

log_transform.fit(X, y)
show_coefficients(log_transform)
