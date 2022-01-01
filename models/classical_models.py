#%%
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
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

#%%
X_train = np.load("../data-clean/X_train_sklearn.npy", allow_pickle=True)
X_test = np.load("../data-clean/X_test_sklearn.npy", allow_pickle=True)
y_train = np.load("../data-clean/y_train_sklearn.npy", allow_pickle=True)
y_test = np.load("../data-clean/y_test_sklearn.npy", allow_pickle=True)

random_state = 123

#%%
# SUBSECTION: Initialize all desired Models
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
    linear_regression,
    lasso,
    ridge,
    decision_tree,
    extra_tree,
    svr,
    ada_boost,
    bagging,
    gradient_boosting,
    random_forest,
]

# SUBSECTION: Save R^2 and MSE for all Models in DataFrame
model_names = []
r2_train_list = []
r2_test_list = []
mse_train_list = []
mse_test_list = []

for model in model_list:
    model_names.append(model.__class__.__name__)
    model.fit(X_train, y_train)

    # performance on training set
    y_hat_train = model.predict(X_train)

    r2_train = r2_score(y_true=y_train, y_pred=y_hat_train)
    r2_train_list.append(r2_train)

    mse_train = mean_squared_error(y_true=y_train, y_pred=y_hat_train)
    mse_train_list.append(mse_train)

    # performance on test set
    y_hat_test = model.predict(X_test)

    r2_test = r2_score(y_true=y_test, y_pred=y_hat_test)
    r2_test_list.append(r2_test)

    mse_test = mean_squared_error(y_true=y_test, y_pred=y_hat_test)
    mse_test_list.append(mse_test)

#%%
# TODO: Results are Nonsense right now, find Mistake in Preprocessing Pipeline
metrics_df = pd.DataFrame(
    {
        "r2_train": r2_train_list,
        "r2_test": r2_test_list,
        "mse_train": mse_train_list,
        "mse_test": mse_test_list,
    },
    index=model_names,
)

metrics_df.sort_values("mse_test")

#%%
