#%%
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# relative imports only work from current directory without package structure
from pytorch_helpers import *

#%%
listings_df = pd.read_pickle("../data-clean/listings.pkl")
listings_df = pd.DataFrame(listings_df)

#%%
IMAGE_SIZE = [224, 224]
batch_size = 8

image_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize(size=IMAGE_SIZE)]
)

#%%
positive_prices = listings_df["price"].loc[listings_df["price"] > 0]
price_bins = positive_prices.quantile(np.linspace(0, 1, 11))

listings_cat_prices_df = (
    listings_df[["price", "picture_url", "host_picture_url"]]
    .dropna()
    .loc[lambda x: x["price"] > 0]
    .assign(
        price_cat=lambda x: pd.cut(x["price"], bins=price_bins, include_lowest=True)
    )
)

# map categories to values 0 - 9 to transform into Tensor
bins_mapping = {
    key: value
    for value, key in dict(
        enumerate(listings_cat_prices_df["price_cat"].unique().sort_values())
    ).items()
}

listings_cat_prices_df = listings_cat_prices_df.assign(
    price_level=lambda x: x["price_cat"].map(bins_mapping).astype("int")
)

#%%
device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")

#%%
class ListingsImages(Dataset):
    def __init__(self, df, image_transforms=None):
        self.x = df["picture_url"]
        self.y = torch.tensor(df["price_level"].values, dtype=torch.long)
        self.image_transforms = image_transforms

    def __getitem__(self, index):
        url = self.x.iloc[index]
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        label = self.y[index]

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        return img, label

    def __len__(self):
        return len(self.y)


#%%
full_dataset = ListingsImages(listings_cat_prices_df, image_transforms)

trainset, valset = generate_train_val_data_split(full_dataset)

trainset, valset = generate_subsets(trainset, valset, subset_size=batch_size)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

#%%
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=24)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


#%%
model = ConvNet().to(device=device)
model

#%%
print_param_shapes(model)

#%%
print_data_shapes(model, device, input_shape=(1, 3, 224, 224))

#%%
lr = 0.01
num_epochs = 5
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

train_losses, val_losses, train_accs, val_accs = run_classification(
    model,
    optimizer,
    loss_function,
    device,
    num_epochs,
    trainloader,
    valloader,
    verbose=True,
)

plot_classification(train_losses, val_losses, train_accs, val_accs)

#%%
