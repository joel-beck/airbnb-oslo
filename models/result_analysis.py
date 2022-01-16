#%%
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn import set_config
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import MLP
from sklearn_helpers import (
    get_column_transformer,
    get_preprocessor,
    print_metrics,
    show_coefficients,
)

simplefilter(action="ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
pd.set_option("precision", 3)
pd.set_option("display.max_columns", 100)
# default value is "text"
set_config(display="diagram")


#%%
# SECTION: Analyze Performance on Validation Set during Training
k_best_results = pd.read_pickle("k_best_results.pkl")
rfe_results = pd.read_pickle("rfe_results.pkl")
vt_results = pd.read_pickle("vt_results.pkl")
pca_results = pd.read_pickle("pca_results.pkl")
full_features_results = pd.read_pickle("full_features_results.pkl")
neural_network_results = pd.read_pickle("neural_network_results.pkl")

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
# Baseline Performance
complete_results.loc["Mean Prediction"]

#%%
complete_results.sort_values("r2_val", ascending=False)


#%%
# stratified by log_y
complete_results.groupby("log_y").apply(lambda x: x.nsmallest(3, "mae_val"))


#%%
complete_results.groupby("log_y").apply(lambda x: x.nlargest(3, "r2_val"))


#%%
plot_data = complete_results.fillna({"feature_selector": "None"}).astype(
    {"feature_selector": "category"}
)[["mae_val", "r2_val", "num_features", "feature_selector", "log_y"]]

g = sns.relplot(
    data=plot_data,
    x="mae_val",
    y=plot_data.index,
    hue="num_features",
    col="feature_selector",
    col_wrap=3,
    style="log_y",
    markers=["s", "o"],
    s=70,
).set(xlabel="", ylabel="", xlim=(380, 550))

g.fig.suptitle("Mean Average Error")
g.fig.subplots_adjust(top=0.9)

sns.move_legend(obj=g, loc="center", bbox_to_anchor=(1, 0.5), frameon=False)


#%%
plot_data = plot_data.sort_values("r2_val", ascending=False)

g = sns.relplot(
    data=plot_data,
    x="r2_val",
    y=plot_data.index,
    hue="num_features",
    col="feature_selector",
    col_wrap=3,
    style="log_y",
    markers=["s", "o"],
    s=70,
).set(xlabel="", ylabel="", xlim=(0.1, 0.35))

g.fig.suptitle(r"$R^2$")
g.fig.subplots_adjust(top=0.9)

sns.move_legend(obj=g, loc="center", bbox_to_anchor=(1, 0.5), frameon=False)


#%%
# SECTION: Evaluate Performance of Best Models on Test Set
X_train_val = pd.read_pickle("../data-clean/X_train_val.pkl")
y_train_val = pd.read_pickle("../data-clean/y_train_val.pkl")

X_test = pd.read_pickle("../data-clean/X_test.pkl")
y_test = pd.read_pickle("../data-clean/y_test.pkl")


#%%
# SUBSECTION: Classical Model with lowest MAE on Validation Set
column_transformer = get_column_transformer()
rfe = RFE(SVR(kernel="linear"), n_features_to_select=30, step=0.5)
preprocessor = get_preprocessor(column_transformer, rfe)

best_model_mae = LinearRegression()
pipeline = make_pipeline(preprocessor, best_model_mae)

# sklearn.set_config(display="diagram") creates nice visual display of pipelines
pipeline

#%%
log_transform = TransformedTargetRegressor(pipeline, func=np.log, inverse_func=np.exp)
log_transform.fit(X_train_val, y_train_val)
y_hat_mae = log_transform.predict(X_test)

print_metrics(y_test, y_hat_mae)
coefs_30 = show_coefficients(log_transform)
coefs_30

#%%
# SUBSECTION: Classical Model with lowest R^2 on Validation Set
rfe = RFE(SVR(kernel="linear"), n_features_to_select=20, step=0.5)
preprocessor = get_preprocessor(column_transformer, rfe)

best_model_r2 = LinearRegression()
pipeline = make_pipeline(preprocessor, best_model_r2)
log_transform = TransformedTargetRegressor(pipeline, func=np.log, inverse_func=np.exp)

log_transform.fit(X_train_val, y_train_val)
y_hat_r2 = log_transform.predict(X_test)

print_metrics(y_test, y_hat_r2)
show_coefficients(log_transform)


#%%
# Neighbourhood has the strongest influence
listings_df = pd.read_pickle("../data-clean/listings_subset.pkl")

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 14))

sns.stripplot(data=listings_df, x="price", y="neighbourhood", ax=ax1).set(
    title="Price Distribution of Neighbourhoods", ylabel=""
)

sns.barplot(data=coefs_30, x="coefficient", y="feature", ax=ax2).set(
    title="Estimated Coefficients of Linear Regression Model", ylabel=""
)

fig.tight_layout()
plt.show()


#%%
# SUBSECTION: Neural Network Model
column_transformer = get_column_transformer()
column_transformer.fit(X_train_val)

X_tensor_test = torch.tensor(column_transformer.transform(X_test).astype(np.float32))
y_tensor_test = torch.tensor(y_test.values.astype(np.float32))
testset = TensorDataset(X_tensor_test, y_tensor_test)
testloader = DataLoader(testset, batch_size=X_tensor_test.shape[0])

in_features = X_tensor_test.shape[1]
hidden_features_list = [64, 128, 256, 512, 512, 256, 128, 64, 32, 16, 8]
dropout_prob = 0.5

model = MLP(in_features, hidden_features_list, dropout_prob)
model.load_state_dict(torch.load("fully_connected_weights.pt"))
model.eval()

with torch.no_grad():
    X_nn, y_nn = next(iter(testloader))
    y_hat_nn = model(X_nn).squeeze()

print_metrics(y_test, y_hat_nn.detach())


#%%
# SUBSECTION: Plot Predictions vs. True Price
best_model_mae_name = best_model_mae.__class__.__name__
best_model_r2_name = best_model_r2.__class__.__name__
size = 20

range_original = [min(y_test), max(y_test)]
range_log = [min(np.log(y_test)), max(np.log(y_test))]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))

ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

# identity line
ax1.scatter(
    y_test.values, y_hat_mae, label=f"{best_model_mae_name}", s=size, color="blue"
)

ax2.scatter(
    np.log(y_test.values),
    np.log(y_hat_mae),
    label=f"{best_model_mae_name}",
    s=size,
    color="blue",
)

ax3.scatter(
    y_test.values, y_hat_r2, label=f"{best_model_r2_name}", s=size, color="green"
)

ax4.scatter(
    np.log(y_test.values),
    np.log(y_hat_r2),
    label=f"{best_model_r2_name}",
    s=size,
    color="green",
)

ax5.scatter(y_test.values, y_hat_nn, label="Neural Network", s=size, color="orange")

ax6.scatter(
    np.log(y_test.values),
    np.log(y_hat_nn),
    label="Neural Network",
    s=size,
    color="orange",
)

for index, ax in enumerate(axes.flat):
    if index % 2 == 0:
        ax.set(ylabel="Predictions")
        ax.legend()
        # identity line
        ax.plot(range_original, range_original, linestyle="dashed", color="grey")
    else:
        ax.plot(range_log, range_log, linestyle="dashed", color="grey")
    if index == 0:
        ax.set(title="Original Scale")
    if index == 1:
        ax.set(title="Log Scale")
    if index in (4, 5):
        ax.set(xlabel="True Price")

fig.tight_layout()
plt.show()

#%%
