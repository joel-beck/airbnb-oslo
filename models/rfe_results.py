#%%
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from matplotlib.ticker import ScalarFormatter
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import MLP
from sklearn_helpers import get_column_transformer, get_preprocessor, show_coefficients

simplefilter(action="ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
pd.set_option("precision", 3)
pd.set_option("display.max_columns", 100)

#%%
# Analysis for Munich or Oslo Data
MUNICH = True

if MUNICH:
    listings_path = "../data-munich/munich_listings.pkl"
    fig_distribution_path = "../term-paper/images/munich_price_distribution.png"
    classical_models_path = "../results-pickle/munich_classical_models_rfe_results.pkl"
    neural_network_path = "../results-pickle/munich_neural_network_rfe_results.pkl"
    complete_results_path = "../results-pickle/munich_complete_rfe_results.pkl"
    fig_comparison_path = "../term-paper/images/munich_model_comparison.png"
    X_train_val_path = "../data-munich/munich_X_train_val.pkl"
    y_train_val_path = "../data-munich/munich_y_train_val.pkl"
    X_test_path = "../data-munich/munich_X_train_val.pkl"
    y_test_path = "../data-munich/munich_y_train_val.pkl"
    fig_coefficient_path = "../term-paper/images/munich_coefficient_plot.png"
    mlp_weights_path = "munich_mlp_weights_None.pt"
    testset_results_path = "../term-paper/tables/munich_table_test_set.csv"
else:
    listings_path = "../data-clean/listings_subset.pkl"
    fig_distribution_path = "../term-paper/images/price_distribution.png"
    classical_models_path = "../results-pickle/classical_models_rfe_results.pkl"
    neural_network_path = "../results-pickle/munich_neural_network_rfe_results.pkl"
    complete_results_path = "../results-pickle/complete_rfe_results.pkl"
    fig_comparison_path = "../term-paper/images/model_comparison.png"
    X_train_val_path = "../data-clean/X_train_val.pkl"
    y_train_val_path = "../data-clean/y_train_val.pkl"
    X_test_path = "../data-clean/X_test_val.pkl"
    y_test_path = "../data-clean/y_test_val.pkl"
    fig_coefficient_path = "../term-paper/images/coefficient_plot.png"
    mlp_weights_path = "mlp_weights_None.pt"
    testset_results_path = "../term-paper/tables/table_test_set.csv"

#%%
# SECTION: Plot Price and Log-Price Distribution
listings_df = pd.read_pickle(listings_path)

fig, axes = plt.subplots(ncols=2, figsize=(10, 6), sharey=True)
ax1, ax2 = axes.flat

sns.histplot(listings_df["price"], ax=ax1).set(
    title="Price Distribution", xlabel="", ylabel=""
)

sns.histplot(listings_df["price"], log_scale=True, ax=ax2).set(
    title="Log-Price Distribution", xlabel="", ylabel=""
)

ax2.xaxis.set_major_formatter(ScalarFormatter())

fig.savefig(fig_distribution_path)
plt.show()


#%%
# SECTION: Model Comparison for different Number of Features
classical_models_rfe_results = pd.read_pickle(classical_models_path)
neural_network_rfe_results = pd.read_pickle(neural_network_path)

complete_rfe_results = pd.concat(
    [
        classical_models_rfe_results,
        neural_network_rfe_results,
    ]
).sort_values("mae_val")

complete_rfe_results.to_pickle(complete_results_path)

#%%
complete_rfe_results

#%%
plot_data = complete_rfe_results.astype({"num_features": "category"})

# sharex="col", sharey="row" is amazing :)
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex="col", sharey=True)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharey=True)
ax1, ax2, ax3, ax4 = axes.flat

sns.scatterplot(
    data=plot_data,
    x="mae_val",
    y=plot_data.index,
    hue="num_features",
    markers=["s", "o"],
    s=70,
    ax=ax1,
    legend=False,
).set(title="MAE Validation", xlabel="")

sns.scatterplot(
    data=plot_data,
    x="mae_train",
    y=plot_data.index,
    hue="num_features",
    markers=["s", "o"],
    s=70,
    ax=ax3,
    legend=False,
).set(title="MAE Training", xlabel="")

sns.scatterplot(
    data=plot_data,
    x="r2_val",
    y=plot_data.index,
    hue="num_features",
    markers=["s", "o"],
    s=70,
    ax=ax2,
    legend=True,
).set(title=r"$R^2$ Validation", xlabel="")

lgd = ax2.legend(title="# Features", bbox_to_anchor=(1.3, 0.2), frameon=False)

sns.scatterplot(
    data=plot_data,
    x="r2_train",
    y=plot_data.index,
    hue="num_features",
    markers=["s", "o"],
    s=70,
    ax=ax4,
    legend=False,
).set(title=r"$R^2$ Training", xlabel="")

sup = fig.suptitle(
    "Model Performances for different Feature Sets\n"
    "Neural Network fitted with prices on original scale, all other models fitted with prices on logarithmic scale"
)

fig.subplots_adjust(top=0.9)
fig.savefig(
    fig_comparison_path,
    bbox_extra_artists=(lgd, sup),
    bbox_inches="tight",
)


#%%
# SUBSECTION: Load Data from Training, Validation and Test Set
X_train_val = pd.read_pickle(X_train_val_path)
y_train_val = pd.read_pickle(y_train_val_path)

X_test = pd.read_pickle(X_test_path)
y_test = pd.read_pickle(y_test_path)

#%%
# SUBSECTION: Coefficient Plot for Linear Regression
column_transformer = get_column_transformer()
rfe = RFE(SVR(kernel="linear"), n_features_to_select=25, step=0.5)
preprocessor = get_preprocessor(column_transformer, rfe)

coef_model = LinearRegression()
pipeline = make_pipeline(preprocessor, coef_model)
log_transform = TransformedTargetRegressor(pipeline, func=np.log, inverse_func=np.exp)

log_transform.fit(X_train_val, y_train_val)
coefs = show_coefficients(log_transform)

#%%
fig, ax = plt.subplots(figsize=(10, 10))

sns.barplot(data=coefs, x="coefficient", y="feature", ax=ax).set(
    title="Estimated Coefficients of Linear Regression Model",
    ylabel="",
    xlabel="Coefficient",
)

fig.savefig(fig_coefficient_path, bbox_inches="tight")

plt.show()

#%%
# SECTION: Evaluate Performance on Test Set for Best Model of each Class
# NOTE: Best Models chosen by lowest MAE
best_models = (
    complete_rfe_results.groupby(complete_rfe_results.index)
    .apply(lambda x: x.nsmallest(1, "mae_val"))
    .sort_values("mae_val")
)

best_models

#%%
# NOTE: Set Hyperparameters for each Model Class manually
# SUBSECTION: Best Linear Regression Model
column_transformer = get_column_transformer()
rfe = RFE(SVR(kernel="linear"), n_features_to_select=50, step=0.5)
preprocessor = get_preprocessor(column_transformer, rfe)

linearreg = LinearRegression()
pipeline = make_pipeline(preprocessor, linearreg)
log_transform = TransformedTargetRegressor(pipeline, func=np.log, inverse_func=np.exp)

log_transform.fit(X_train_val, y_train_val)
linearreg_predictions = log_transform.predict(X_test)

#%%
# SUBSECTION: Best Ridge Model
column_transformer = get_column_transformer()
rfe = RFE(SVR(kernel="linear"), n_features_to_select=50, step=0.5)
preprocessor = get_preprocessor(column_transformer, rfe)

ridge = Ridge(alpha=14)
pipeline = make_pipeline(preprocessor, ridge)
log_transform = TransformedTargetRegressor(pipeline, func=np.log, inverse_func=np.exp)

log_transform.fit(X_train_val, y_train_val)
ridge_predictions = log_transform.predict(X_test)

#%%
# SUBSECTION: Best Random Forest Model
column_transformer = get_column_transformer()
randomforest = RandomForestRegressor(n_estimators=6, min_samples_leaf=4, max_depth=5)

pipeline = make_pipeline(column_transformer, randomforest)
log_transform = TransformedTargetRegressor(pipeline, func=np.log, inverse_func=np.exp)

log_transform.fit(X_train_val, y_train_val)
randomforest_predictions = log_transform.predict(X_test)

#%%
# SUBSECTION: Best HistGradientBoosting Model
column_transformer = get_column_transformer()
histgradientboosting = HistGradientBoostingRegressor(
    min_samples_leaf=4, max_leaf_nodes=30, max_iter=40, max_depth=18, learning_rate=0.09
)

pipeline = make_pipeline(column_transformer, histgradientboosting)
log_transform = TransformedTargetRegressor(pipeline, func=np.log, inverse_func=np.exp)

log_transform.fit(X_train_val, y_train_val)
histgradientboosting_predictions = log_transform.predict(X_test)

#%%
# SUBSECTION: Best Neural Network Model
column_transformer = get_column_transformer()
column_transformer.fit(X_train_val)

X_tensor_test = torch.tensor(column_transformer.transform(X_test).astype(np.float32))
y_tensor_test = torch.tensor(y_test.values.astype(np.float32))
testset = TensorDataset(X_tensor_test, y_tensor_test)
testloader = DataLoader(testset, batch_size=X_tensor_test.shape[0], shuffle=False)

in_features = X_tensor_test.shape[1]
hidden_features_list = [64, 128, 256, 128, 64, 8]
dropout_prob = 0.5
use_skip_connections = True

mlp = MLP(in_features, hidden_features_list, dropout_prob, use_skip_connections)
mlp.load_state_dict(torch.load(mlp_weights_path))
mlp.eval()

with torch.no_grad():
    X_nn, y_nn = next(iter(testloader))
    mlp_predictions = mlp(X_nn).squeeze()

#%%
# SUBSECTION: Add Ensemble Predictions
top2_average = np.mean([ridge_predictions, linearreg_predictions], axis=0)

top3_average = np.mean(
    [ridge_predictions, linearreg_predictions, histgradientboosting_predictions], axis=0
)

top4_average = np.mean(
    [
        ridge_predictions,
        linearreg_predictions,
        histgradientboosting_predictions,
        randomforest_predictions,
    ],
    axis=0,
)

top5_average = np.mean(
    [
        ridge_predictions,
        linearreg_predictions,
        histgradientboosting_predictions,
        randomforest_predictions,
        mlp_predictions.detach().numpy(),
    ],
    axis=0,
)

#%%
# SECTION: Evaluate All Predictions on Test Set
df_index = [
    "Linear Regression",
    "Ridge",
    "Random Forest",
    "HistGradientBoosting",
    "Neural Network",
    "Top2 Average",
    "Top3 Average",
    "Top4 Average",
    "Top5 Average",
]

testset_df_list = []

for predictions in [
    linearreg_predictions,
    ridge_predictions,
    randomforest_predictions,
    histgradientboosting_predictions,
    mlp_predictions.detach().numpy(),
    top2_average,
    top3_average,
    top4_average,
    top5_average,
]:

    temp_df = pd.DataFrame(
        {
            "MAE": [mean_absolute_error(y_test, predictions)],
            "R2": [r2_score(y_test, predictions)],
            "MSE": [mean_squared_error(y_test, predictions)],
        }
    )
    testset_df_list.append(temp_df)

testset_df = pd.concat(testset_df_list)
testset_df.index = df_index

testset_df

#%%
testset_df.drop(columns=["MSE"]).round(3).to_csv(testset_results_path)

#%%
