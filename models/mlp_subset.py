#%%
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import MLP, run_regression
from sklearn_helpers import ResultContainer, get_column_transformer, get_preprocessor

#%%
def get_data_subset(
    X_train_val: pd.DataFrame, y_train_val: pd.DataFrame, num_features: int
) -> tuple[TensorDataset, TensorDataset]:
    # 58 transformed columns
    column_transformer = get_column_transformer()
    rfe = RFE(SVR(kernel="linear"), n_features_to_select=num_features, step=0.5)
    preprocessor = get_preprocessor(column_transformer, rfe)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=123, shuffle=True
    )

    X_train_tensor = torch.tensor(
        preprocessor.fit_transform(X_train, y_train).astype(np.float32)
    )
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
    trainset = TensorDataset(X_train_tensor, y_train_tensor)

    X_val_tensor = torch.tensor(preprocessor.transform(X_val).astype(np.float32))
    y_val_tensor = torch.tensor(y_val.values.astype(np.float32))
    valset = TensorDataset(X_val_tensor, y_val_tensor)

    return trainset, valset


#%%
# SECTION: PyTorch Training Test
X_train_val = pd.read_pickle("../data-clean/X_train_val.pkl")
y_train_val = pd.read_pickle("../data-clean/y_train_val.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# BOOKMARK: Hyperparameters
hidden_features_list = [64, 128, 256, 512, 512, 256, 128, 64, 32, 16, 8]

batch_size = 128
num_epochs = 200
dropout_prob = 0.5

log_y_list = [True, False]
num_features_list = [1, 2, 5, 10, 25, 50]

#%%
log_y = False
neural_network_results = []

for num_features in num_features_list:
    trainset, valset = get_data_subset(X_train_val, y_train_val, num_features)
    in_features = len(trainset[0][0])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    model = MLP(in_features, hidden_features_list, dropout_prob).to(device)

    loss_function = nn.MSELoss()
    lr = 0.1 if log_y else 0.01
    optimizer = Adam(params=model.parameters(), lr=lr)

    result_container = ResultContainer(
        model_names=["NeuralNetwork"], feature_selector=["rfe"]
    )
    # append actual number of observed features
    result_container.num_features.append(in_features)
    result_container.hyperparam_keys.append(
        ["batch_size", "num_epochs", "dropout_probability"]
    )
    result_container.hyperparam_values.append([batch_size, num_epochs, dropout_prob])

    # save model weights only for log_y = False
    save_path = None if log_y else "fully_connected_weights.pt"

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
        # save_path=save_path,
    )

    metrics.plot()
    neural_network_results.append(result_container.display_df())

pd.concat(neural_network_results).sort_values("mae_val").to_pickle(
    "../results-pickle/neural_network_subsets.pkl"
)

#%%
# already good performance for the following 10 included features:
column_transformer = get_column_transformer()
rfe = RFE(SVR(kernel="linear"), n_features_to_select=10, step=0.5)
preprocessor = get_preprocessor(column_transformer, rfe)
preprocessor.fit_transform(X_train_val, y_train_val)

encoded_features = preprocessor.named_steps[
    "column_transformer"
].get_feature_names_out()

selected_features = preprocessor.named_steps["feature_selector"].get_feature_names_out(
    encoded_features
)

print("Selected Features from RFE:")
for feature in selected_features:
    print(feature.split("__")[1])

#%%
