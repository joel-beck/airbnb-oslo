#%%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from pytorch_helpers import (
    plot_regression,
    print_data_shapes,
    print_param_shapes,
    run_regression,
    generate_train_val_data_split,
    init_data_loaders,
)
from sklearn_helpers import get_column_transformer

os.chdir("/Users/marei/airbnb-oslo")

#%%
# SECTION: PyTorch Training Test
listings_subset = pd.read_pickle("data-clean/listings_subset.pkl")

X = listings_subset.drop(columns="price")
y = listings_subset["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, shuffle=True
)

column_transformer = get_column_transformer()

X_train_tensor = torch.tensor(
    column_transformer.fit_transform(X_train).astype(np.float32)
)
y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
trainset = TensorDataset(X_train_tensor, y_train_tensor)

X_test_tensor = torch.tensor(column_transformer.transform(X_test).astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.float32))
testset = TensorDataset(X_test_tensor, y_test_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# BOOKMARK: Hyperparameters
batch_size = 64

# NN structure
in_features = len(trainset[0][0])
hidden_features_list = [32, 64, 128, 64, 32]
dropout_prob = 0.25
loss_function = nn.MSELoss(reduction="sum")

#%%
# BOOKMARK: Generate train-val split

trainset, valset = generate_train_val_data_split(trainset, split_seed=42, val_frac=0.2)

#%%
# BOOKMARK: DataLoaders

trainloader, valloader, testloader = init_data_loaders(
    trainset, valset, testset, batch_size
)

#%%
# SECTION: Model Construction
class NN(nn.Module):
    def __init__(self, in_features, hidden_features_list, dropout_prob):
        super(NN, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features_list[0], bias=True),
            # nn.BatchNorm1d(hidden_features_list[0]),
            nn.ReLU(),
            # nn.Dropout(dropout_prob),
        )

        self.hidden_layers = self.hidden_block(
            in_features=hidden_features_list[0],
            out_features_list=hidden_features_list[1:],
            dropout_prob=dropout_prob,
        )

        self.output_layer = nn.Linear(
            in_features=hidden_features_list[-1], out_features=1
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def hidden_block(self, in_features, out_features_list, dropout_prob):
        layers = []
        for out_features in out_features_list:
            layers.append(nn.Linear(in_features, out_features, bias=True))
            # layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(p=dropout_prob))
            in_features = out_features

        return nn.Sequential(*layers)


model = NN(in_features, hidden_features_list, dropout_prob).to(device)

#%%
print("Parameter shapes")
print_param_shapes(model)

#%%
# SECTION: Model Training
# model = NN(in_features, hidden_features_list, dropout_prob).to(device)

num_epochs = 50
lr = 0.01
optimizer = optim.Adam(params=model.parameters(), lr=lr)

# NOTE: Adjusted return values to display mean absolute error and r2
train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s = run_regression(
    model,
    optimizer,
    loss_function,
    device,
    num_epochs,
    trainloader,
    testloader,
    verbose=True,
    save_best=True,
)

plot_regression(train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s)

#%%
