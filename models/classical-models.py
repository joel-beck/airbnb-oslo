#%%
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from time import perf_counter

# model fitting at the end produces
# FutureWarning: Arrays of bytes/strings is being converted to decimal numbers if
# dtype='numeric'.
# This behavior is deprecated in 0.24 and will be removed in 1.1 (renaming of 0.26).
# Please convert your data to numeric values explicitly instead.
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

#%%
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

X = listings_subset.drop(columns="price")
y = listings_subset["price"]

#%%
# SECTION: Transform Categorical Columns to Dummies and Standardize Numeric Columns
# NOTE: This section of the code is currently duplicated with the pytorch_preprocessing file => maybe find solution to remove that redundancy
categorical_cols = [
    "host_gender",
    "host_identity_verified",
    "host_is_superhost",
    "neighbourhood",
    "room_type",
    "shared_bathrooms",
]

numeric_cols = [
    col
    for col in listings_subset.columns
    if col not in categorical_cols and col != "price"
]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_cols),
    (categorical_transformer, categorical_cols),
)

random_state = 123

#%%
# SUBSECTION: Initialize all desired Models
# right now naive implementation without specifying/tuning hyperparameters
linear_regression = LinearRegression()
lasso = Lasso(random_state=random_state)
ridge = Ridge(random_state=random_state)
decision_tree = DecisionTreeRegressor(random_state=random_state)
extra_tree = ExtraTreeRegressor(random_state=random_state)
svr = SVR()
ada_boost = AdaBoostRegressor(random_state=random_state)
bagging = BaggingRegressor(random_state=random_state)
gradient_boosting = GradientBoostingRegressor(random_state=random_state)
random_forest = RandomForestRegressor(random_state=random_state)

#%%
model_list = [
    ada_boost,
    bagging,
    # simple trees massively overfit training data
    # decision_tree,
    # extra_tree,
    gradient_boosting,
    lasso,
    linear_regression,
    random_forest,
    ridge,
    # does not learn anything
    # svr,
]

# store means of all ten folds for each model and each metric in list
model_names = []
r2_train_list = []
r2_test_list = []
mse_train_list = []
mse_test_list = []

#%%
# SECTION: Combine Preprocessing, Model Fitting and Cross-Validation Evaluation
start = perf_counter()

for model in model_list:
    print("Fitting", model.__class__.__name__)

    # preprocessing steps and model fit (same preprocessing here inefficiently
    # duplicated)
    pipeline = make_pipeline(preprocessor, model)

    # scores returns a dictionary with all specified metrics for train and test set,
    # get all available metrics with
    # import sklearn
    # sorted(sklearn.metrics.SCORERS.keys())
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=10,
        scoring=["r2", "neg_mean_squared_error"],
        return_train_score=True,
    )

    model_names.append(model.__class__.__name__)

    r2_train_list.append(np.mean(scores["train_r2"]))
    r2_test_list.append(np.mean(scores["test_r2"]))

    # for some reason, only negative mean squared error is an available metric
    mse_train_list.append(-np.mean(scores["train_neg_mean_squared_error"]))
    mse_test_list.append(-np.mean(scores["test_neg_mean_squared_error"]))

print(f"Finished training in {perf_counter() - start: .2f} seconds")

#%%
# SUBSECTION: Analyze Results
metrics_df = pd.DataFrame(
    {
        "r2_train": r2_train_list,
        "r2_test": r2_test_list,
        "mse_train": mse_train_list,
        "mse_test": mse_test_list,
    },
    index=model_names,
)

# NOTE: Performances terrible right now => probably design mistake
metrics_df.sort_values("r2_test", ascending=False)

#%%
