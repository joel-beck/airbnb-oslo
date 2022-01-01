#%%
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset

#%%
listings_subset = pd.read_pickle("listings_subset.pkl")

#%%
# SECTION: Transform Categorical Columns to Dummies and Standardize Numeric Columns
categorical_cols = [
    "host_gender",
    "host_identity_verified",
    "host_is_superhost",
    "neighbourhood",
    "room_type",
    "shared_bathrooms",
]

numeric_cols = [
    col
    for col in listings_subset.columns
    if col not in categorical_cols and col != "price"
]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_cols),
        ("categorical", categorical_transformer, categorical_cols),
    ]
)

#%%
X = listings_subset.drop(columns="price")
y = listings_subset["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, shuffle=True
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#%%
# SUBSECTION: Scikit-Learn Training Test
np.save(arr=X_train, file="X_train_sklearn.npy", allow_pickle=True)
np.save(arr=X_test, file="X_test_sklearn.npy", allow_pickle=True)
np.save(arr=y_train, file="y_train_sklearn.npy", allow_pickle=True)
np.save(arr=y_test, file="y_test_sklearn.npy", allow_pickle=True)

#%%
# SUBSECTION: PyTorch Training Test
X_train = torch.tensor(X_train.astype(np.float32))
y_train = torch.tensor(y_train.values.astype(np.float32))
trainset = TensorDataset(X_train, y_train)

X_test = torch.tensor(X_test.astype(np.float32))
y_test = torch.tensor(y_test.values.astype(np.float32))
testset = TensorDataset(X_test, y_test)

torch.save(trainset, "trainset_pytorch.pt")
torch.save(testset, "testset_pytorch.pt")

#%%
