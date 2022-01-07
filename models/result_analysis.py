#%%
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import LinearRegression
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
y_hat_classic = pipeline.predict(X)

mean_absolute_error(y, y_hat_classic), r2_score(y, y_hat_classic)

#%%
# SUBSECTION: Fit and Predict with Neural Network Model
column_transformer = get_column_transformer()
X_tensor = torch.tensor(column_transformer.fit_transform(X).astype(np.float32))
y_tensor = torch.tensor(y.values.astype(np.float32))
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=X_tensor.shape[0])

in_features = X_tensor.shape[1]
hidden_features_list = [64, 128, 256, 512, 512, 256, 128, 64, 32, 16, 8]
dropout_prob = 0.5

model = LinearRegression(in_features, hidden_features_list, dropout_prob)
model.load_state_dict(torch.load("fully_connected_weights.pt"))
model.eval()

with torch.no_grad():
    X_nn, y_nn = next(iter(dataloader))
    y_hat_nn = model(X_nn).squeeze()

mean_absolute_error(y, y_hat_nn.detach()), r2_score(y, y_hat_nn.detach())

#%%
# BOOKMARK: Predictions vs. True Price

sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

# identity line
ax1.plot(
    [min(y), max(y)],
    [min(y), max(y)],
    linestyle="dashed",
    color="grey",
)

ax1.scatter(
    y.values, y_hat_classic, label=f"{best_model.__class__.__name__}", alpha=0.5
)
ax1.scatter(y.values, y_hat_nn, label="Neural Network", alpha=0.5)
ax1.set(title="Original Scale", xlabel="True Price", ylabel="Predictions")
ax1.legend()

# identity line
ax2.plot(
    [min(np.log(y)), max(np.log(y))],
    [min(np.log(y)), max(np.log(y))],
    linestyle="dashed",
    color="grey",
)

ax2.scatter(
    np.log(y.values),
    np.log(y_hat_classic),
    label=f"{best_model.__class__.__name__}",
    alpha=0.5,
)
ax2.scatter(np.log(y.values), np.log(y_hat_nn), label="Neural Network", alpha=0.5)
ax2.set(title="Log Scale", xlabel="True Price", ylabel="Predictions")

fig.tight_layout()
sns.move_legend(
    obj=ax1, loc="upper center", bbox_to_anchor=(1.1, 1.2), ncol=2, frameon=False
)

sns.despine()
plt.show()

#%%
