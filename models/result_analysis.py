#%%
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

from sklearn_helpers import get_column_transformer, get_preprocessor

simplefilter(action="ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
pd.set_option("precision", 3)

#%%
# SUBSECTION: Collect Results
k_best_results = pd.read_pickle("k_best_results.pkl")
rfe_results = pd.read_pickle("rfe_results.pkl")
vt_results = pd.read_pickle("vt_results.pkl")
pca_results = pd.read_pickle("pca_results.pkl")
full_features_results = pd.read_pickle("full_features_results.pkl")
neural_network_results = pd.read_pickle("neural_network_results.pkl")

#%%
complete_results = pd.concat(
    [
        k_best_results,
        rfe_results,
        vt_results,
        pca_results,
        full_features_results,
        neural_network_results,
    ]
).sort_values("mae_val")

complete_results.to_pickle("complete_results.pkl")

#%%
# SUBSECTION: Analyze Results
complete_results

#%%
complete_results.sort_values("r2_val", ascending=False)

#%%
plot_data = (
    complete_results.fillna({"feature_selector": "None"})
    .astype({"feature_selector": "category"})[
        ["mae_val", "r2_val", "num_features", "feature_selector"]
    ]
    .loc[lambda x: x.index != "Mean Prediction"]
)

g = sns.relplot(
    data=plot_data,
    x="mae_val",
    y=plot_data.index,
    hue="num_features",
    col="feature_selector",
    col_wrap=3,
    s=70,
).set(xlabel="", ylabel="", xlim=(380, 550))

g.fig.suptitle("Mean Average Error")
g.fig.subplots_adjust(top=0.9)

#%%
plot_data = plot_data.sort_values("r2_val", ascending=False)

g = sns.relplot(
    data=plot_data,
    x="r2_val",
    y=plot_data.index,
    hue="num_features",
    col="feature_selector",
    col_wrap=3,
    s=70,
).set(xlabel="", ylabel="", xlim=(0.1, 0.3))

g.fig.suptitle("R^2")
g.fig.subplots_adjust(top=0.9)

#%%
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

X = listings_subset.drop(columns="price")
y = listings_subset["price"]

#%%
# BOOKMARK: Price Distribution
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 6))
sns.histplot(y, ax=ax1).set(title="Original Scale")
sns.histplot(y, log_scale=True, ax=ax2).set(title="Log Scale")
plt.show()

#%%
# SUBSECTION: Fit and Predict with Best Classical Model
column_transformer = get_column_transformer()
rfe = RFE(SVR(kernel="linear"), n_features_to_select=10, step=0.5)
preprocessor = get_preprocessor(column_transformer, rfe)
best_model = RandomForestRegressor(n_estimators=6, min_samples_leaf=7, max_depth=4)

pipeline = make_pipeline(preprocessor, best_model)
pipeline.fit(X, y)
y_hat = pipeline.predict(X)

mean_absolute_error(y, y_hat), r2_score(y, y_hat)

#%%
# BOOKMARK: Predictions vs. True Price

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 6))
ax1.plot(
    [min(y), max(y)],
    [min(y), max(y)],
    linestyle="dashed",
    color="grey",
)
pd.DataFrame({"True Price": y.values, "Predictions": y_hat}).plot(
    kind="scatter", x="True Price", y="Predictions", ax=ax1
)
ax1.set(title="Original Scale")

ax2.plot(
    [min(np.log(y)), max(np.log(y))],
    [min(np.log(y)), max(np.log(y))],
    linestyle="dashed",
    color="grey",
)
pd.DataFrame({"True Price": np.log(y.values), "Predictions": np.log(y_hat)}).plot(
    kind="scatter", x="True Price", y="Predictions", ax=ax2
)
ax2.set(title="Log Scale")

plt.show()

#%%
