#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import MLP, run_regression
from sklearn_helpers import ResultContainer, get_column_transformer

pd.set_option("precision", 3)

#%%
X_train_val = pd.read_pickle("../data-clean/X_train_val.pkl")
y_train_val = pd.read_pickle("../data-clean/y_train_val.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
def get_data(
    X_train_val: pd.DataFrame, y_train_val: pd.DataFrame, quantile_threshold: float
) -> tuple[TensorDataset, TensorDataset]:
    # drop all observations above certain quantile in price distribution
    keep_index = y_train_val.loc[
        y_train_val <= y_train_val.quantile(1 - (quantile_threshold / 100))
    ].index

    y_train_val = y_train_val.loc[keep_index]
    X_train_val = X_train_val.loc[keep_index]

    # currently 59 transformed columns
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
num_epochs = 200
dropout_prob = 0.5
use_skip_connections = True
lr = 0.01
log_y = False

quantile_threshold_list = [0, 1, 2.5, 5, 10]

#%%
# SUBSECTION: Fit Neural Network
quantile_threshold_results = []

for quantile_threshold in quantile_threshold_list:
    trainset, valset, column_transformer = get_data(
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

output_df = pd.concat(quantile_threshold_results).sort_values("mae_val")


#%%
cols = list(output_df.columns)
cols = [cols[-1]] + cols[:-1]
output_df = output_df[cols]

output_df.to_pickle("../results-pickle/neural_network_outliers.pkl")
