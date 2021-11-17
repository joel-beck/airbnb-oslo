#%%
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_utils import (
    plot_results,
    print_data_shapes,
    print_param_shapes,
    run_training,
)
from torch.utils.data import DataLoader, Subset

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainset = torch.load("../data-clean/listings_train.pt")
valset = torch.load("../data-clean/listings_val.pt")

#%%
# hyperparameters
batch_size = 64
in_features = len(trainset[0][0])
num_hidden_layers = 2
hidden_features = 64
loss_function = nn.MSELoss()

#%%
# comment out to train on whole dataset
# subset_size = batch_size * 2

# train_indices = torch.randint(0, len(trainset) + 1, size=(subset_size,))
# trainset = Subset(dataset=trainset, indices=train_indices)

# val_indices = torch.randint(0, len(valset) + 1, size=(subset_size,))
# valset = Subset(dataset=valset, indices=val_indices)

#%%
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

#%%
class LinearRegression(nn.Module):
    def __init__(self, in_features, hidden_features, num_hidden_layers):
        super(LinearRegression, self).__init__()
        self.input_layer = nn.Linear(
            in_features=in_features, out_features=hidden_features
        )
        hidden_block = [
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
        ]
        self.hidden_layers = nn.Sequential(*(num_hidden_layers * hidden_block))
        self.output_layer = nn.Linear(in_features=hidden_features, out_features=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


model = LinearRegression(in_features, hidden_features, num_hidden_layers)
model

#%%
print_param_shapes(model)

#%%
print_data_shapes(model, device, input_shape=(1, in_features))

#%%
model = LinearRegression(in_features, hidden_features, num_hidden_layers)

num_epochs = 50
lr = 0.001
optimizer = optim.Adam(params=model.parameters(), lr=lr)

train_losses, val_losses = run_training(
    model,
    optimizer,
    loss_function,
    device,
    num_epochs,
    trainloader,
    valloader,
    verbose=True,
)

plot_results(train_losses, val_losses)

#%%
