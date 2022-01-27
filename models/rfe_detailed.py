#%%
from warnings import simplefilter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import MLP, print_param_shapes, run_regression
from sklearn_helpers import (
    ResultContainer,
    fit_models,
    get_column_transformer,
    get_models,
    get_preprocessor,
)

simplefilter(action="ignore", category=FutureWarning)
pd.set_option("precision", 3)

#%%
X_train_val = pd.read_pickle("../data-clean/X_train_val.pkl")
y_train_val = pd.read_pickle("../data-clean/y_train_val.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# SECTION: Neural Network
def get_data_subset(
    X_train_val: pd.DataFrame, y_train_val: pd.DataFrame, num_features: int
) -> tuple[TensorDataset, TensorDataset]:
    # currently 59 transformed columns
    column_transformer = get_column_transformer()

    if num_features is not None:
        rfe = RFE(SVR(kernel="linear"), n_features_to_select=num_features, step=0.5)
        preprocessor = get_preprocessor(column_transformer, rfe)
    else:
        preprocessor = column_transformer

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

    return trainset, valset, preprocessor


#%%
# BOOKMARK: Hyperparameters

# hidden_features_list = [64, 256, 1024, 256, 64, 8]
hidden_features_list = [64, 128, 256, 128, 64, 8]

batch_size = 128
num_epochs = 200
dropout_prob = 0.5
use_skip_connections = True

lr = 0.01
scheduler_rate = 0.5
scheduler_patience = int(num_epochs / 10)
log_y = False

num_features_list = [None, 50, 25, 10, 5, 2, 1]

#%%
# SUBSECTION: Fit Neural Network
neural_network_results = []

for i, num_features in enumerate(num_features_list):
    trainset, valset, preprocessor = get_data_subset(
        X_train_val, y_train_val, num_features
    )
    in_features = len(trainset[0][0])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    model = MLP(
        in_features, hidden_features_list, dropout_prob, use_skip_connections
    ).to(device)

    if i == 0:
        print_param_shapes(model)

    loss_function = nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_rate, patience=scheduler_patience
    )

    result_container = ResultContainer(
        model_names=["NeuralNetwork"], feature_selector=["RFE"]
    )
    # append actual number of observed features
    result_container.num_features.append(in_features)
    result_container.selected_features.append(
        [feature.split("__")[1] for feature in preprocessor.get_feature_names_out()]
    )
    result_container.hyperparam_keys.append(
        ["batch_size", "num_epochs", "dropout_probability"]
    )
    result_container.hyperparam_values.append([batch_size, num_epochs, dropout_prob])

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
        # scheduler=scheduler,
        verbose=True,
        save_best=True,
    )

    metrics.plot()
    neural_network_results.append(result_container.display_df())

pd.concat(neural_network_results).sort_values("mae_val").to_pickle(
    "../results-pickle/neural_network_rfe_results.pkl"
)

#%%
# SECTION: Classical Models

# BOOKMARK: Hyperparameters
random_state = 42
n_folds = 10
n_iter = 10
log_y = True

num_features_list = [None, 50, 25, 10, 5, 2, 1]

#%%
# SUBSECTION: Fit Classical Models
column_transformer = get_column_transformer()
classical_model_results = []

for num_features in num_features_list:

    if num_features is not None:
        rfe = RFE(SVR(kernel="linear"), n_features_to_select=num_features, step=0.5)
        preprocessor = get_preprocessor(column_transformer, rfe)
    else:
        preprocessor = Pipeline([("column_transformer", column_transformer)])

    models = get_models(
        preprocessor,
        models=["linear", "lasso", "ridge", "random_forest", "hist_gradient_boosting"],
        random_state=random_state,
        log_y=log_y,
    )

    result_container = ResultContainer()

    result = fit_models(
        X_train_val,
        y_train_val,
        models,
        result_container,
        n_folds,
        n_iter,
        random_state,
        log_y=log_y,
    )

    classical_model_results.append(result.display_df())

pd.concat(classical_model_results).sort_values("mae_val").to_pickle(
    "../results-pickle/classical_models_rfe_results.pkl"
)


#%%
