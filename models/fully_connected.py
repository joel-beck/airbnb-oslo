#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import (
    LinearRegression,
    print_data_shapes,
    print_param_shapes,
    run_regression,
)
from sklearn_helpers import ResultContainer, get_column_transformer

#%%
# SECTION: PyTorch Training Test
listings_subset = pd.read_pickle("../data-clean/listings_subset.pkl")

X = listings_subset.drop(columns="price")
y = listings_subset["price"]

result_container = ResultContainer(
    model_names=["NeuralNetwork"], feature_selector=[None]
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=123, shuffle=True
)

column_transformer = get_column_transformer()

X_train_tensor = torch.tensor(
    column_transformer.fit_transform(X_train).astype(np.float32)
)
y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
trainset = TensorDataset(X_train_tensor, y_train_tensor)
result_container.num_features.append(X_train_tensor.shape[1])

X_val_tensor = torch.tensor(column_transformer.transform(X_val).astype(np.float32))
y_val_tensor = torch.tensor(y_val.values.astype(np.float32))
valset = TensorDataset(X_val_tensor, y_val_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# BOOKMARK: Hyperparameters
in_features = len(trainset[0][0])
hidden_features_list = [64, 128, 256, 512, 512, 256, 128, 64, 32, 16, 8]

batch_size = 128
num_epochs = 100
dropout_prob = 0.5
lr = 0.01

result_container.grid_key_list.append(
    ["batch_size", "num_epochs", "learning_rate", "dropout_probability"]
)
result_container.grid_value_list.append([batch_size, num_epochs, lr, dropout_prob])

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
model = LinearRegression(in_features, hidden_features_list, dropout_prob).to(device)
model

#%%
print_param_shapes(model)

#%%
# print_data_shapes(model, device, input_shape=(1, in_features))

#%%
# SECTION: Model Training
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
    verbose=True,
    save_best=True,
    save_path="fully_connected_weights.pt",
)

metrics.plot_results()

#%%
metrics_df = result_container.display_results()
metrics_df.to_pickle("neural_network_results.pkl")
metrics_df
