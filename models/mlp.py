#%%
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import MLP, print_param_shapes, run_regression
from sklearn_helpers import ResultContainer, get_column_transformer

#%%
# SECTION: PyTorch Training Test
X_train_val = pd.read_pickle("../data-clean/X_train_val.pkl")
y_train_val = pd.read_pickle("../data-clean/y_train_val.pkl")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=123, shuffle=True
)

column_transformer = get_column_transformer()

#%%
X_train_tensor = torch.tensor(
    column_transformer.fit_transform(X_train).astype(np.float32)
)
y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
trainset = TensorDataset(X_train_tensor, y_train_tensor)

X_val_tensor = torch.tensor(column_transformer.transform(X_val).astype(np.float32))
y_val_tensor = torch.tensor(y_val.values.astype(np.float32))
valset = TensorDataset(X_val_tensor, y_val_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# BOOKMARK: Hyperparameters
in_features = len(trainset[0][0])
hidden_features_list = [64, 128, 256, 512, 512, 256, 128, 64, 32, 16, 8]

batch_size = 128
num_epochs = 200
dropout_prob = 0.5
scheduler_rate = 0.5
scheduler_patience = int(num_epochs / 5)

log_y_list = [True, False]
weight_decay_list = [0, 0.01, 0.1, 1]  # L2 penalty; no weight decay: set value to 0

#%%
# BOOKMARK: DataLoaders
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

#%%
# SECTION: Model Construction
model = MLP(in_features, hidden_features_list, dropout_prob).to(device)
model

#%%
print_param_shapes(model)

#%%
# SECTION: Model Training

#%%
# NOTE: This Cell can be run to get a feeling for the performance of various combinations of learning rate schedulers and values for the weight decay. Based on my experience the Adam optimizer actually works best in our case with no weight decay and no learning rate scheduler.

# set to True to include cell in script
TRY_PARAM_COMBINATIONS = False

if TRY_PARAM_COMBINATIONS:
    neural_network_results = []

    for (log_y, weight_decay) in itertools.product(log_y_list, weight_decay_list):
        model = MLP(in_features, hidden_features_list, dropout_prob).to(device)

        loss_function = nn.MSELoss()
        lr = 0.1 if log_y else 0.01
        optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

        # try different learning rate schedulers
        # cannot be defined on top, since they depend on the optimizer in this iteration
        step_lr = StepLR(optimizer, step_size=scheduler_patience, gamma=scheduler_rate)
        reduce_plateau = ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_rate, patience=scheduler_patience
        )
        scheduler_list = [None, step_lr, reduce_plateau]

        for scheduler in scheduler_list:
            # initialize result container in each iteration
            result_container = ResultContainer(
                model_names=["NeuralNetwork"], feature_selector=[None]
            )
            result_container.num_features.append(X_train_tensor.shape[1])
            result_container.hyperparam_keys.append(
                ["batch_size", "num_epochs", "dropout_probability"]
            )
            result_container.hyperparam_values.append(
                [batch_size, num_epochs, dropout_prob]
            )

            # save model weights only for log_y = False
            # save_path = None if log_y else "fully_connected_weights.pt"

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
                scheduler=scheduler,
                verbose=True,
                save_best=True,
                # save_path=save_path,
            )

            metrics.plot()
            neural_network_results.append(result_container.display_df())

    pd.concat(
        neural_network_results
    )  # .to_pickle("neural_network_results_regularized.pkl")

#%%
neural_network_results = []

for log_y in log_y_list:
    model = MLP(in_features, hidden_features_list, dropout_prob).to(device)

    loss_function = nn.MSELoss()
    lr = 0.1 if log_y else 0.01
    optimizer = Adam(params=model.parameters(), lr=lr)

    result_container = ResultContainer(
        model_names=["NeuralNetwork"], feature_selector=[None]
    )
    result_container.num_features.append(X_train_tensor.shape[1])
    result_container.hyperparam_keys.append(
        ["batch_size", "num_epochs", "dropout_probability"]
    )
    result_container.hyperparam_values.append([batch_size, num_epochs, dropout_prob])

    # save model weights only for log_y = False
    save_path = None if log_y else "mlp_weights.pt"

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
        save_path=save_path,
    )

    metrics.plot()
    neural_network_results.append(result_container.display_df())

pd.concat(neural_network_results).to_pickle(
    "../results-pickle/neural_network_results.pkl"
)

#%%
