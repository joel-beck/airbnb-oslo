#%%
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn import functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from sklearn_helpers import get_column_transformer

sns.set_theme(style="whitegrid")


#%%
# SECTION: PyTorch Training Test
X_train_val = pd.read_pickle("../data-clean/X_train_val.pkl")
y_train_val = pd.read_pickle("../data-clean/y_train_val.pkl")

quantiles = [0, 0.25, 0.9, 0.99, 1]

tensor_labels = np.arange(1, len(quantiles))
plot_labels = ["0-25", "25-90", "90-99", ">99"]
label_mapping = dict(zip(tensor_labels, plot_labels))

bins = y_train_val.quantile(quantiles)
y_categories = pd.cut(y_train_val, bins, labels=tensor_labels, include_lowest=True)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_categories, test_size=0.2, random_state=123, shuffle=True
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
batch_size = 128
num_epochs = 2000
learning_rate = 0.01
latent_space_dim = 2

#%%
# BOOKMARK: DataLoaders
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

#%%
# SUBSECTION: Calculate mean and std for each column in training set
# temploader = DataLoader(trainset, batch_size=len(trainset))
# x, y = next(iter(temploader))

# data_means = x.mean(dim=0, keepdim=True)
# data_stds = x.std(dim=0, keepdim=True)

#%%
# SECTION: Define VAE
class VAE(nn.Module):
    def __init__(self, in_features, latent_space_dim):
        super(VAE, self).__init__()

        self.enc_fc1 = nn.Linear(in_features, 32)
        self.enc_fc2 = nn.Linear(32, 16)
        self.enc_fc3 = nn.Linear(16, 8)
        self.enc_fc4 = nn.Linear(8, 4)
        self.relu = nn.ReLU()

        self.mu = nn.Linear(4, latent_space_dim)
        self.logvar = nn.Linear(4, latent_space_dim)
        self.z = nn.Linear(latent_space_dim, 4)

        self.dec_fc1 = nn.Linear(4, 8)
        self.dec_fc2 = nn.Linear(8, 16)
        self.dec_fc3 = nn.Linear(16, 32)
        self.dec_fc4 = nn.Linear(32, in_features)

        self.data_means = data_means
        self.data_stds = data_stds

    def encode(self, x):
        # input shape: (N, 59)
        x = self.enc_fc1(x)  # shape: (N, 32)
        x = self.relu(x)
        x = self.enc_fc2(x)  # shape: (N, 16)
        x = self.relu(x)
        x = self.enc_fc3(x)  # shape: (N, 8)
        x = self.relu(x)
        x = self.enc_fc4(x)  # shape: (N, 4)
        x = self.relu(x)

        mu = self.mu(x)  # shape: (N, 2)
        logvar = self.logvar(x)  # shape: (N, 2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # std = var^0.5 = exp(log(var^0.5)) = exp(0.5 * log(var))
        std = torch.exp(0.5 * logvar)
        # random sample from standard normal distribution
        eps = torch.randn_like(std)
        # random sample from normal distribution with mean mu and variance var = std^2 = exp(logvar)
        z = mu + std * eps
        return z

    def decode(self, z):
        # input shape: (N, 2)
        x = self.z(z)  # shape: (N, 4)
        x = self.relu(x)
        x = self.dec_fc1(x)  # shape: (N, 8)
        x = self.relu(x)
        x = self.dec_fc2(x)  # shape: (N, 16)
        x = self.relu(x)
        x = self.dec_fc3(x)  # shape: (N, 32)
        x = self.relu(x)
        x = self.dec_fc4(x)  # shape: (N, 59)

        # standardize to bring in similar range to input data
        mean_vec = x.mean(dim=0, keepdim=True)
        std_vec = x.std(dim=0, keepdim=True)
        recon_x = (x - mean_vec) / std_vec

        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


#%%
# SUBSECTION: VAE Loss Function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction losses are calculated using Mean Squared Error (MSE) and
    # summed over all elements and batch
    mse_loss = F.mse_loss(input=recon_x, target=x, reduction="sum")
    kld_loss = 0.5 * torch.sum(mu ** 2 + torch.exp(logvar) - logvar - 1)

    total_loss = mse_loss + kld_loss
    return total_loss


#%%
# SUBSECTION: Define Training Loop
def train(
    model,
    trainloader,
    optimizer,
    loss_function,
    device,
):
    batch_losses = []
    model.train()

    # use only the images without their labels, train_loader provides both
    for x, _ in trainloader:
        x = x.to(device=device)

        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)

        loss = loss_function(recon_x, x, mu, logvar)
        batch_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        return np.mean(batch_losses)


def validate(model, valloader, loss_function, device):
    batch_losses = []
    model.eval()
    with torch.no_grad():
        for x, _ in valloader:
            x = x.to(device=device)
            recon_x, mu, logvar = model(x)

            loss = loss_function(recon_x, x, mu, logvar)
            batch_losses.append(loss.item())

            return np.mean(batch_losses)


def run_training(
    model,
    trainloader,
    valloader,
    optimizer,
    loss_function,
    num_epochs,
    device,
    verbose=True,
):
    start_time = perf_counter()

    train_losses = []
    test_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, trainloader, optimizer, loss_function, device)
        test_loss = validate(model, valloader, loss_function, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if verbose:
            if epoch % int(num_epochs / 5) == 0:
                print(f"Epoch: {epoch} / {num_epochs}\n{'-' * 50}")
                print(
                    f"Epoch Train Loss: {train_loss:.3f} | Epoch Validation Loss: {test_loss:.3f}\n"
                )

    time_elapsed = np.round(perf_counter() - start_time, 0).astype(int)

    if verbose:
        print(f"Finished training after {time_elapsed} seconds.")

    return train_losses, test_losses


def plot(train_losses, test_losses):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Training Loss", marker="o")
    ax.plot(epochs, test_losses, label="Validation Loss", marker="o")
    ax.set(xlabel="Epoch", ylabel="")
    ax.legend()

    sns.move_legend(
        obj=ax, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False
    )
    sns.despine()
    plt.show()


#%%
# SECTION: Training
model = VAE(in_features, latent_space_dim).to(device=device)
optimizer = Adam(model.parameters(), lr=learning_rate)
model

#%%
train_losses, val_losses = run_training(
    model,
    trainloader,
    valloader,
    optimizer,
    loss_function=vae_loss,
    num_epochs=num_epochs,
    device=device,
    verbose=True,
)

plot(train_losses, val_losses)

#%%
example = torch.rand(size=(32, 59))
encoding = model.encode(example)
z = model.reparameterize(*encoding)
model.decode(z)

#%%
# SECTION: Plot Latent Space Representation
# SUBSECTION: Encode entire Training Set
trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)

x, y = next(iter(trainloader))
x = x.to(device=device)

with torch.no_grad():
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar).detach().cpu().numpy()

#%%
# SUBSECTION: Plot Latent Space, color by Price Category
plot_data = pd.DataFrame({"z1": z[:, 0], "z2": z[:, 1], "price_cat": y})

plot_data["Price Quantiles"] = (
    plot_data["price_cat"].map(label_mapping).astype("category")
)

g = sns.relplot(
    data=plot_data,
    x="z1",
    y="z2",
    hue="Price Quantiles",
    size="Price Quantiles",
    sizes=[30, 30, 30, 200],
    height=10,
    aspect=1,
).set(xlabel="Latent Dimension 1", ylabel="Latent Dimension 2")

g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Feature Representation in 2-Dimensional Latent Space")
g.fig.savefig("../term-paper/images/latent_representation.png")

sns.move_legend(g, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.95))

#%%
