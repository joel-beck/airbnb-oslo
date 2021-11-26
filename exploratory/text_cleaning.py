import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
import torchtext
#from torchtext.data import Field, TabularDataset, BucketIterator

#%%
listings = pd.read_pickle("listings.pkl")

reviews = pd.read_pickle("reviews.pkl")

reviews_subset = reviews[0:99]
reviews_subset = reviews_subset.join(listings['price'], reviews_subset.index)

reviews_df = reviews_subset.loc[:, ['comments', 'price']]
x_train = reviews_subset.loc[:, 'comments']
y_train = reviews_subset.loc[:, 'price']


#%%

tokenize = lambda x: x.split()

comment = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
price = Field(sequential=False, use_vocab=False)

fields = {'comment': ('c', comment), 'price': ('p', price)}


