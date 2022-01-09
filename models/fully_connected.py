#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import (
    MLP,
    print_data_shapes,
    print_param_shapes,
    run_regression,
    init_weights,
)
from sklearn_helpers import ResultContainer, get_column_transformer

#%%
# SECTION: PyTorch Training Test
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

X = listings_subset.drop(columns="price")
y = listings_subset["price"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=123, shuffle=True
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
lr = 0.01
log_y = False

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
# print_data_shapes(model, device, input_shape=(1, in_features))

#%%
# SECTION: Model Training
neural_network_results = []

for log_y in [True, False]:
    result_container = ResultContainer(
        model_names=["NeuralNetwork"], feature_selector=[None]
    )
    result_container.num_features.append(X_train_tensor.shape[1])
    result_container.hyperparam_keys.append(
        ["batch_size", "num_epochs", "learning_rate", "dropout_probability"]
    )
    result_container.hyperparam_values.append(
        [batch_size, num_epochs, lr, dropout_prob]
    )

    model = MLP(in_features, hidden_features_list, dropout_prob).to(device)

    # avoid negative predicted prices at beginning of training to enable log transformation
    if log_y:
        model.apply(lambda x: init_weights(x, mean=50))

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

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
        # save_path="fully_connected_weights.pt",
    )

    metrics.plot()
    neural_network_results.append(result_container.display_df())

pd.concat(neural_network_results).to_pickle("neural_network_results.pkl")
