#%%
from io import BytesIO

import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

# relative imports only work from current directory without package structure
from pytorch_helpers import (
    generate_subsets,
    generate_train_val_data_split,
    plot_regression,
    print_param_shapes,
    run_regression,
)

#%%
front_page_pictures = pd.read_pickle("../data-clean/front_page_pictures.pkl")
listings_df = pd.read_pickle("../data-clean/listings.pkl")

#%%
# drop prices of 0, largest price (outlier) and missing images
picture_price_df = (
    pd.merge(
        front_page_pictures, listings_df["price"], left_index=True, right_index=True
    )
    .loc[lambda x: (x["price"] > 0) & (x["price"] < 80000)]
    .dropna()
)

#%%
IMAGE_SIZE = [224, 224]
batch_size = 8

image_transforms = transforms.Compose(
    [transforms.Resize(size=IMAGE_SIZE), transforms.PILToTensor()]
)

# normalization requires four dimensions, done after unsqueezing
# values of channel means and standard deviations from documentation for resnet https://pytorch.org/vision/stable/models.html
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#%%
device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")

#%%
class ListingsImages(Dataset):
    """
    Creates PyTorch Dataset from Pandas DataFrame containing (at least) one Column of Picture URLs and one additional Column of corresponding Prices.
    The resulting Images are resized and normalized four-dimensional Tensors and serve, together with the returned Price Tensors, as Input to a PyTorch DataLoader object.
    """

    def __init__(self, df, image_transforms=None):
        self.x = df["listing_url"]
        self.y = torch.tensor(df["price"].values, dtype=torch.float)
        self.image_transforms = image_transforms

    def __getitem__(self, index):
        url = self.x.iloc[index]
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        label = self.y[index]

        if self.image_transforms is not None:
            img_tensor = self.image_transforms(img).to(dtype=torch.float)
            img_tensor = normalize(img_tensor)

        return img_tensor, label

    def __len__(self):
        return len(self.y)


#%%
full_dataset = ListingsImages(picture_price_df, image_transforms)
trainset, valset = generate_train_val_data_split(full_dataset)

# comment out to train on full dataset
trainset, valset = generate_subsets(trainset, valset, subset_size=batch_size)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

#%%
model = resnet18(pretrained=True).to(device=device)

# freeze weights
for param in model.parameters():
    param.requires_grad = False

# replace last fully connected layer, weights of new layer require gradient computation
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)

print_param_shapes(model)

#%%
params_to_update = [param for param in model.parameters() if param.requires_grad]
print("Parameters to train:", sum(param.numel() for param in params_to_update))

#%%
lr = 0.01
num_epochs = 5
# use only parameters with requires_grad = True in optimizer
optimizer = optim.Adam(params_to_update, lr=lr)

loss_function = nn.MSELoss()
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
