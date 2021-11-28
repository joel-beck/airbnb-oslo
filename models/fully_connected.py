#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pytorch_helpers import (
    plot_regression,
    print_data_shapes,
    print_param_shapes,
    run_regression,
)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainset = torch.load("../data-clean/listings_train.pt")
valset = torch.load("../data-clean/listings_val.pt")

#%%
# hyperparameters
batch_size = 64
in_features = len(trainset[0][0])
hidden_features_list = [32, 64, 128, 64, 32]
dropout_prob = 0.5
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
    def __init__(self, in_features, hidden_features_list, dropout_prob):
        super(LinearRegression, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features_list[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
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
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            in_features = out_features

        return nn.Sequential(*layers)


model = LinearRegression(in_features, hidden_features_list, dropout_prob).to(device)
model

#%%
print_param_shapes(model)

#%%
print_data_shapes(model, device, input_shape=(1, in_features))

#%%
model = LinearRegression(in_features, hidden_features_list, dropout_prob).to(device)

num_epochs = 200
lr = 0.001
optimizer = optim.Adam(params=model.parameters(), lr=lr)

train_losses, val_losses = run_regression(
    model,
    optimizer,
    loss_function,
    device,
    num_epochs,
    trainloader,
    valloader,
    verbose=True,
)

plot_regression(train_losses, val_losses)

#%%
