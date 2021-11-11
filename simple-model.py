#%%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

#%%
listings_df = pd.read_pickle("data-clean/listings.pkl")

#%%
numeric_cols = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
    "host_acceptance_rate",
    "number_bathrooms",
    "bedrooms",
    "review_scores_rating",
]

listings_subset = (
    listings_df[numeric_cols].dropna().apply(lambda x: (x - x.mean()) / x.std())
)

#%%
y_data = torch.tensor(listings_subset["price"].values.astype(np.float32))
X_data = torch.tensor(listings_subset.drop(columns=["price"]).values.astype(np.float32))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
subset_size = 10
batch_size = 5

#%%
train_data = TensorDataset(X_data, y_data)

# comment out to train on whole dataset
train_indices = torch.randint(0, len(train_data) + 1, size=(subset_size,))
train_data = Subset(dataset=train_data, indices=train_indices)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

#%%
def print_param_shapes(model, col_widths=(25, 8)):
    for name, param in model.named_parameters():
        print(
            f"Name: {name:<{col_widths[0]}} | # Params: {param.numel():<{col_widths[1]}} | Shape: {list(param.shape)}"
        )
    print("\nTotal number of parameters:", sum(p.numel() for p in model.parameters()))


def _print_shape(input, layer=None, col_width=25):
    if layer is None:
        print(f"{f'Input shape:':<{col_width}} {list(input.shape)}")
    else:
        print(f"{f'{layer.__class__.__name__} output shape:':<25} {list(input.shape)}")


def print_data_shapes(model, input_shape):
    x = torch.rand(size=input_shape, dtype=torch.float32)
    _print_shape(x)

    for i, layer in enumerate(model.modules()):
        if i == 0:
            continue

        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                x = sub_layer(x)
                _print_shape(x, sub_layer)
        else:
            x = layer(x)
            _print_shape(x, layer)


def train_epoch(dataloader, optimizer, model, loss_fn, device):
    epoch_loss, epoch_total = 0.0, 0.0

    for X, y in dataloader:
        X = X.to(device=device)
        y = y.to(device=device)

        optimizer.zero_grad()
        model.train()

        y_pred = model(X)

        batch_size = len(y)
        epoch_total += batch_size

        # loss per sample
        loss = loss_fn(y_pred, y)
        # Loss per minibatch
        epoch_loss += loss * batch_size

        loss.backward()
        optimizer.step()

    return epoch_loss.detach().to(device="cpu").numpy() / epoch_total


def run_training(
    model,
    optimizer,
    loss_function,
    device,
    num_epochs,
    train_dataloader,
    verbose=False,
):
    start_time = time.time()
    train_losses = []

    for epoch in range(1, num_epochs + 1):

        epoch_train_loss = train_epoch(
            dataloader=train_dataloader,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_function,
            device=device,
        )

        train_losses.append(epoch_train_loss)

        if verbose:
            if epoch % int(num_epochs / 5) == 0:
                print(f"Epoch: {epoch} / {num_epochs}\n{'-' * 50}")
                print(f"Mean Loss Training: {epoch_train_loss:.5f}")

    time_elapsed = np.round(time.time() - start_time, 0).astype(int)
    print(f"Finished training after {time_elapsed} seconds.")

    return train_losses


def plot_results(train_losses):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Training")
    ax.set(
        title="Average Train Loss per Observation",
        xlabel="Epoch",
        ylabel="",
    )
    ax.legend()

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    sns.despine()
    plt.show()


#%%
in_features = X_data.shape[1]
hidden_features = 64
loss_function = nn.MSELoss()

#%%
class LinearRegression(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


model = LinearRegression(in_features, hidden_features)
model

#%%
# X, y = next(iter(train_loader))
# model(X)

#%%
print_param_shapes(model)
print()
print_data_shapes(model, input_shape=(1, in_features))

#%%
num_epochs = 2000
lr = 0.1
optimizer = optim.Adam(params=model.parameters(), lr=lr)

#%%
train_losses = run_training(
    model, optimizer, loss_function, device, num_epochs, train_loader, verbose=True
)

#%%
# cannot overfit 10 training samples => network design flawed :/
plot_results(train_losses)

#%%
