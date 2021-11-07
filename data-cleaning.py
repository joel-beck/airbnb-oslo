# %%
import os

import geopandas as gpd
import pandas as pd

# %%
# Overview over all Datasets
dir = "data-raw"
files = os.listdir(dir)

for file in files:
    if file.endswith("json"):
        df = gpd.read_file("/".join([dir, file]))
    else:
        df = pd.read_csv("/".join([dir, file]))
    print(f"File: {file}, Shape: {df.shape}")
    print(df.iloc[:3, :5], "\n")


# %% [markdown]
# # Data Cleaning

# %% [markdown]
# ## Merge / Select DataFrames

# %%
# SECTION: Data Cleaning
# SUBSECTION: Merge / Select DataFrames
# combine two neighbourhood dataframes
nbhood_1 = pd.read_csv("data-raw/neighbourhoods.csv")
nbhood_2 = gpd.read_file("data-raw/neighbourhoods.geojson")

neighbourhoods_df = pd.merge(
    nbhood_1.drop(columns=["neighbourhood_group"]),
    nbhood_2.drop(columns=["neighbourhood_group"]),
    how="left",
)

# %%
# reviews.csv redundant => keep only reviews.csv.gz
reviews_df = pd.read_csv("./data-raw/reviews.csv.gz", parse_dates=["date"])

# %%
list_1 = pd.read_csv("data-raw/listings.csv")
list_2 = pd.read_csv("data-raw/listings.csv.gz")

list_1.shape, list_2.shape

# only neighbourhood_group column (which only contains NA's) in listings.csv but not in listings.csv.gz
print(list_1.columns[~list_1.columns.isin(list_2.columns)])
print(list_1["neighbourhood_group"].isna().all())

# contain same row information => keep only larger dataframe
print((list_1["id"] == list_2["id"]).all())

listings_df = list_2

# %%
calendar_df = pd.read_csv("data-raw/calendar.csv.gz", parse_dates=["date"])

# %% [markdown]
# Keep only 4 out of 7 Data Frames:
# - listings.csv.gz
# - calendar.csv.gz
# - reviews.csv.gz
# - neighbourhoods.geojson

# %%
# all three dataframes are connected by id columns
print(listings_df["id"].nunique())
print(calendar_df["listing_id"].nunique())
print(reviews_df["listing_id"].nunique())

# %%
# listings_df strict superset of unique observations
print(calendar_df["listing_id"].isin(listings_df["id"]).all())
print(reviews_df["listing_id"].isin(listings_df["id"]).all())

# calendar_df not a superset of reviews_df observations
print(reviews_df["listing_id"].isin(calendar_df["listing_id"]).all())

# %% [markdown]
# ## Convert Data Types

# %%
# SUBSECTION: Convert Data Types
print(reviews_df.dtypes, "\n")

reviews_df = reviews_df.convert_dtypes()
print(reviews_df.dtypes)

# %%
neighbourhoods_df = neighbourhoods_df.convert_dtypes()
print(neighbourhoods_df.dtypes)

# %%
print(calendar_df.dtypes, "\n")

calendar_df = calendar_df.convert_dtypes().assign(
    available=calendar_df["available"].astype("category"),
    price=calendar_df["price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype("float"),
    adjusted_price=calendar_df["adjusted_price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype("float"),
)
print(calendar_df.dtypes)

# %%
# change first_review and last_review to date and price to float
for col in listings_df.convert_dtypes():
    print(col, "\t", listings_df[col].dtype)

listings_df = listings_df.assign(
    first_review=pd.to_datetime(listings_df["first_review"]),
    last_review=pd.to_datetime(listings_df["last_review"]),
    price=listings_df["price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype("float"),
)

listings_df[["price", "first_review", "last_review"]].dtypes

# %%
# Write clean Datasets to file
listings_df.to_pickle(path="data-clean/listings.pkl")
reviews_df.to_pickle(path="data-clean/reviews.pkl")
calendar_df.to_pickle(path="data-clean/calendar.pkl")
neighbourhoods_df.to_pickle(path="data-clean/neighbourhoods.pkl")
