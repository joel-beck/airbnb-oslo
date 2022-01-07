#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import (
    plot_regression,
    print_data_shapes,
    print_param_shapes,
    run_regression,
)
from sklearn_helpers import get_column_transformer

#%%
# SECTION: PyTorch Training Test
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

X = listings_subset.drop(columns="price")
y = listings_subset["price"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=123, shuffle=True
)

column_transformer = get_column_transformer()

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
batch_size = 128
in_features = len(trainset[0][0])
hidden_features_list = [64, 128, 256, 512, 512, 256, 128, 64, 32, 16, 8]
dropout_prob = 0.5
loss_function = nn.MSELoss()

#%%
# BOOKMARK: Subset
# comment out to train on whole dataset
# subset_size = batch_size * 2

# train_indices = torch.randint(0, len(trainset) + 1, size=(subset_size,))
# trainset = Subset(dataset=trainset, indices=train_indices)

# val_indices = torch.randint(0, len(testset) + 1, size=(subset_size,))
# testset = Subset(dataset=testset, indices=val_indices)

#%%
# BOOKMARK: DataLoaders
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

#%%
# SECTION: Model Construction
class LinearRegression(nn.Module):
    def __init__(self, in_features, hidden_features_list, dropout_prob):
        super(LinearRegression, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features_list[0], bias=False),
            nn.BatchNorm1d(hidden_features_list[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
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
            layers.append(nn.Linear(in_features, out_features, bias=False))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            in_features = out_features

        return nn.Sequential(*layers)


model = LinearRegression(in_features, hidden_features_list, dropout_prob).to(device)
model

#%%
print_param_shapes(model)

#%%
# print_data_shapes(model, device, input_shape=(1, in_features))

#%%
# SECTION: Model Training
model = LinearRegression(in_features, hidden_features_list, dropout_prob).to(device)

num_epochs = 100
lr = 0.01
optimizer = optim.Adam(params=model.parameters(), lr=lr)

train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s = run_regression(
    model,
    optimizer,
    loss_function,
    device,
    num_epochs,
    trainloader,
    valloader,
    verbose=True,
    save_best=True,
)

plot_regression(train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s)

#%%
