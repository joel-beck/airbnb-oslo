#%%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

#%%
listings_df = pd.read_pickle("listings.pkl")
reviews_features = pd.read_pickle("reviews_features.pkl")

#%%
# SECTION: Column Selection & Preprocessing
# SUBSECTION: Choose Subset with most important columns

# not including host_acceptance_rate increases rows without missing values in
# listings_processed from 1978 to 2481
listings_cols = [
    "price",
    "neighbourhood",
    "room_type",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
    # "host_acceptance_rate",
    "host_is_superhost",
    "number_bathrooms",
    "shared_bathrooms",
    "bedrooms",
    "review_scores_rating",
]

reviews_cols = [
    "median_review_length",
    "number_languages",
    "frac_norwegian",
]

# add numeric features from reviews dataframe to listings_subset,
# join() merges by index
listings_subset = listings_df[listings_cols].join(reviews_features[reviews_cols])
listings_subset.to_pickle("listings_subset.pkl")

#%%
# SUBSECTION: Transform Categorical Columns to Dummies
categorical_cols = [
    "neighbourhood",
    "room_type",
    "host_is_superhost",
    "shared_bathrooms",
]

# exclude observations with price of 0
listings_processed = (
    pd.get_dummies(listings_subset, columns=categorical_cols, drop_first=True)
    .loc[listings_df["price"] > 0]
    .dropna()
)

listings_processed.to_pickle("listings_processed.pkl")

#%%
# SECTION: Split in Training and Test Set
# SUBSECTION: Standardize Numeric Columns based on Training Set
def standardize(df, numeric_cols, train_indices):
    df = df.copy()
    mean_vec = df[numeric_cols].iloc[train_indices].mean()
    std_vec = df[numeric_cols].iloc[train_indices].std()
    df[numeric_cols] = (df[numeric_cols] - mean_vec) / std_vec
    return df


rng = np.random.default_rng(seed=123)
train_frac = 0.8
train_indices = rng.choice(
    range(len(listings_processed)),
    size=int(train_frac * len(listings_processed)),
    replace=False,
)
val_indices = [
    index for index in range(0, len(listings_processed)) if index not in train_indices
]
numeric_cols = [col for col in listings_subset.columns if col not in categorical_cols]

listings_standardized = standardize(listings_processed, numeric_cols, train_indices)

#%%
# SUBSECTION: Pandas Training Test
listings_train = listings_standardized.iloc[train_indices]
listings_val = listings_standardized.iloc[val_indices]

listings_train.to_pickle("listings_train_pandas.pkl")
listings_val.to_pickle("listings_val_pandas.pkl")

#%%
# SUBSECTION: PyTorch Training Test
y_train = torch.tensor(listings_train["price"].values.astype(np.float32))
X_train = torch.tensor(listings_train.drop(columns=["price"]).values.astype(np.float32))
trainset = TensorDataset(X_train, y_train)

y_val = torch.tensor(listings_val["price"].values.astype(np.float32))
X_val = torch.tensor(listings_val.drop(columns=["price"]).values.astype(np.float32))
valset = TensorDataset(X_val, y_val)

torch.save(trainset, "listings_train_pytorch.pt")
torch.save(valset, "listings_val_pytorch.pt")

#%%
