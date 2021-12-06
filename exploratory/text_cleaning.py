import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
import torchtext
#from torchtext.data import Field, TabularDataset, BucketIterator

#%%
listings_df = pd.read_pickle("data-clean/listings.pkl")

reviews_df = pd.read_pickle("data-clean/reviews.pkl")

reviews = reviews_df.join(listings_df['price'], reviews_df.index)

reviews = reviews.loc[:, ['comments', 'price']]
x_train = reviews.loc[:, 'comments']
y_train = reviews.loc[:, 'price']


#%%

# transform to PyTorch Dataset
#x_train = torch.tensor(np.array(x_train))
#y_train = torch.tensor(y_train)
#trainset = TensorDataset(x_train, y_train)

#torch.save(trainset, "text_train.pt")

 #%%
#tokenize = lambda x: x.split()

#text = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
#price = LabelField(sequential=False, use_vocab=False)

#fields = {'text': ('c', text), 'price': ('p', price)}



#%%

# create price categories

positive_prices = reviews["price"].loc[reviews["price"] > 0]
price_bins = positive_prices.quantile(np.linspace(0, 1, 11))

reviews_cat_prices = (
    reviews.dropna()
    .loc[lambda x: x["price"] > 0]
    .assign(
        price_cat=lambda x: pd.cut(x["price"], bins=price_bins, include_lowest=True)
    )
)

# map categories to values 0 - 9 to transform into Tensor
bins_mapping = {
    key: value
    for value, key in dict(
        enumerate(reviews_cat_prices["price_cat"].unique().sort_values())
    ).items()
}

reviews_cat_prices = reviews_cat_prices.assign(
    price_level=lambda x: x["price_cat"].map(bins_mapping).astype("int")
)

#%%

# create DataSet class

class Reviews_Text(Dataset):
    def __init__(self, df, transform=None):
        self.x = df["comments"]
        self.y = torch.tensor(df["price_level"].values, dtype=torch.long)
        self.transform = transform

    def __getitem__(self, idx):
        comment = self.x[idx]
        label = self.y[idx]

        return comment, label

    def __len__(self):
        return len(self.y)

#%%

dataset = Reviews_Text(reviews_cat_prices)


#%%

