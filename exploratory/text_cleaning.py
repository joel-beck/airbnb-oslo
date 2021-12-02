import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
import torchtext
#from torchtext.data import Field, TabularDataset, BucketIterator

#%%
listings = pd.read_pickle("data-clean/listings.pkl")

reviews = pd.read_pickle("data-clean/reviews.pkl")

reviews_subset = reviews[0:99]
reviews_subset = reviews_subset.join(listings['price'], reviews_subset.index)

reviews_df = reviews_subset.loc[:, ['comments', 'price']]
x_train = reviews_subset.loc[:, 'comments']
y_train = reviews_subset.loc[:, 'price']


#%%

# transform to PyTorch Dataset
#x_train = torch.tensor(np.array(x_train))
y_train = torch.tensor(y_train)
trainset = TensorDataset(x_train, y_train)

torch.save(trainset, "text_train.pt")

 #%%
tokenize = lambda x: x.split()

text = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
price = LabelField(sequential=False, use_vocab=False)

fields = {'text': ('c', text), 'price': ('p', price)}



#%%

