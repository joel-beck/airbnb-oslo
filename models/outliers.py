#%%
from warnings import simplefilter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import MLP, run_regression
from sklearn_helpers import (
    ResultContainer,
    fit_models,
    get_column_transformer,
    get_models,
)

simplefilter(action="ignore", category=FutureWarning)
pd.set_option("precision", 3)
pd.set_option("display.max_columns", 100)

#%%
# Analysis for Munich or Oslo Data
MUNICH = True

if MUNICH:
    X_train_val_path = "../data-munich/munich_X_train_val.pkl"
    y_train_val_path = "../data-munich/munich_y_train_val.pkl"
    mlp_results_path = "../results-pickle/munich_neural_network_outliers.pkl"
    classical_models_results_path = (
        "../results-pickle/munich_classical_models_outliers.pkl"
    )
    table_outliers_path = "../term-paper/tables/munich_table_outliers.csv"
    outlier_results_path = "../results-pickle/munich_outlier_results.pkl"
else:
    X_train_val_path = "../data-clean/X_train_val.pkl"
    y_train_val_path = "../data-clean/y_train_val.pkl"
    mlp_results_path = "../results-pickle/neural_network_outliers.pkl"
    classical_models_results_path = "../results-pickle/classical_models_outliers.pkl"
    table_outliers_path = "../term-paper/tables/table_outliers.csv"
    outlier_results_path = "../results-pickle/outlier_results.pkl"


#%%
X_train_val = pd.read_pickle(X_train_val_path)
y_train_val = pd.read_pickle(y_train_val_path)

# look into outliers (here largest quantile of prices)
y_train_val.loc[y_train_val > y_train_val.quantile(1 - (1 / 100))]
X_train_val.loc[y_train_val > y_train_val.quantile(1 - (1 / 100))]

#%%
# SECTION: Neural Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
def get_mlp_data(
    X_train_val: pd.DataFrame, y_train_val: pd.Series, quantile_threshold: float
) -> tuple[TensorDataset, TensorDataset, ColumnTransformer]:
    # drop all observations above certain quantile in price distribution
    keep_index = y_train_val.loc[
        y_train_val <= y_train_val.quantile(1 - (quantile_threshold / 100))
    ].index

    y_train_val = y_train_val.loc[keep_index]
    X_train_val = X_train_val.loc[keep_index]

    # 59 transformed columns for Oslo / 92 transformed columns for Munich
    column_transformer = get_column_transformer()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=123, shuffle=True
    )

    X_train_tensor = torch.tensor(
        column_transformer.fit_transform(X_train, y_train).astype(np.float32)
    )
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
    trainset = TensorDataset(X_train_tensor, y_train_tensor)

    X_val_tensor = torch.tensor(column_transformer.transform(X_val).astype(np.float32))
    y_val_tensor = torch.tensor(y_val.values.astype(np.float32))
    valset = TensorDataset(X_val_tensor, y_val_tensor)

    return trainset, valset, column_transformer


#%%
# BOOKMARK: Hyperparameters
hidden_features_list = [64, 128, 256, 128, 64, 8]

batch_size = 128
num_epochs = 300
dropout_prob = 0.5
use_skip_connections = True
lr = 0.01
log_y = False

quantile_threshold_list = [0, 1, 2.5, 5, 10]

#%%
# SUBSECTION: Fit Neural Network
quantile_threshold_results = []

for quantile_threshold in quantile_threshold_list:
    trainset, valset, column_transformer = get_mlp_data(
        X_train_val, y_train_val, quantile_threshold
    )
    in_features = len(trainset[0][0])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    model = MLP(
        in_features, hidden_features_list, dropout_prob, use_skip_connections
    ).to(device)

    loss_function = nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=lr)

    result_container = ResultContainer(
        model_names=["NeuralNetwork"], feature_selector=[None]
    )
    # append actual number of observed features
    result_container.num_features.append(in_features)
    result_container.selected_features.append(
        [
            feature.split("__")[1]
            for feature in column_transformer.get_feature_names_out()
        ]
    )
    result_container.hyperparam_keys.append(
        ["batch_size", "num_epochs", "dropout_probability"]
    )
    result_container.hyperparam_values.append([batch_size, num_epochs, dropout_prob])

    print(f"Quantile Threshold: {quantile_threshold}\n")

    metrics, result_container = run_regression(
        model,
        optimizer,
        loss_function,
        device,
        num_epochs,
        trainloader,
        valloader,
        result_container,
        log_y=log_y,
        verbose=True,
        save_best=True,
    )

    metrics.plot()

    df = result_container.display_df().assign(quantile_threshold=quantile_threshold)
    quantile_threshold_results.append(df)

mlp_results = pd.concat(quantile_threshold_results).sort_values("mae_val")


#%%
cols = list(mlp_results.columns)
cols = [cols[-1]] + cols[:-1]
mlp_results = mlp_results[cols]

mlp_results.to_pickle(mlp_results_path)

#%%
# SECTION: Classical Models
def get_classical_data(
    X_train_val: pd.DataFrame, y_train_val: pd.Series, quantile_threshold: float
) -> tuple[pd.DataFrame, pd.Series]:
    # drop all observations above certain quantile in price distribution
    keep_index = y_train_val.loc[
        y_train_val <= y_train_val.quantile(1 - (quantile_threshold / 100))
    ].index

    y_train_val = y_train_val.loc[keep_index]
    X_train_val = X_train_val.loc[keep_index]

    return X_train_val, y_train_val


#%%
# BOOKMARK: Hyperparameters
random_state = 42
n_folds = 10
n_iter = 10
log_y = True

quantile_threshold_list = [0, 1, 2.5, 5, 10]

#%%
# SUBSECTION: Fit Classical Models
X_train_val = pd.read_pickle(X_train_val_path)
y_train_val = pd.read_pickle(y_train_val_path)

column_transformer = get_column_transformer()
preprocessor = Pipeline([("column_transformer", column_transformer)])
quantile_threshold_results = []

for quantile_threshold in quantile_threshold_list:

    X, y = get_classical_data(X_train_val, y_train_val, quantile_threshold)
    models = get_models(
        preprocessor,
        models=["linear", "ridge", "random_forest", "hist_gradient_boosting"],
        random_state=random_state,
        log_y=log_y,
    )

    result_container = ResultContainer()

    result = fit_models(
        X,
        y,
        models,
        result_container,
        n_folds,
        n_iter,
        random_state,
        log_y=log_y,
    )

    df = result.display_df().assign(quantile_threshold=quantile_threshold)
    quantile_threshold_results.append(df)

classical_models_results = pd.concat(quantile_threshold_results).sort_values("mae_val")

#%%
cols = list(classical_models_results.columns)
cols = [cols[-1]] + cols[:-1]
classical_models_results = classical_models_results[cols]

classical_models_results.to_pickle(classical_models_results_path)

#%%
mlp_results = pd.read_pickle(mlp_results_path)

table_outliers = (
    mlp_results[["quantile_threshold", "mae_val", "r2_val"]]
    .rename(
        columns={
            "quantile_threshold": "Quantile Threshold",
            "mae_val": "Mean Absolute Error",
            "r2_val": "R2",
        }
    )
    .set_index("Quantile Threshold")
    .sort_index()
    .round(2)
)

table_outliers.to_csv(table_outliers_path)

#%%
classical_models_results = pd.read_pickle(classical_models_results_path)

outlier_results = pd.concat([mlp_results, classical_models_results]).sort_values(
    "mae_val"
)

outlier_results.to_pickle(outlier_results_path)

#%%
