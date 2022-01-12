#%%
import os

# conda/mamba install geopandas
import geopandas as gpd
import pandas as pd

#%%
# SECTION: Merge / Select DataFrames
# combine two neighbourhood dataframes
nbhood_1 = pd.read_csv("../data-raw/neighbourhoods.csv")
nbhood_2 = gpd.read_file("../data-raw/neighbourhoods.geojson")

neighbourhoods_df = pd.merge(
    nbhood_1,
    nbhood_2.drop(columns=["neighbourhood_group"]),
    how="left",
)

#%%
# reviews.csv redundant => keep only reviews.csv.gz
reviews_df = pd.read_csv(
    "../data-raw/reviews.csv.gz", parse_dates=["date"], index_col="listing_id"
)

#%%
list_1 = pd.read_csv("../data-raw/listings.csv", index_col="id")
list_2 = pd.read_csv("../data-raw/listings.csv.gz", index_col="id")

# only neighbourhood_group column (which only contains NA's) in listings.csv but not in listings.csv.gz, however contents of common columns are different
print(list_1.columns[~list_1.columns.isin(list_2.columns)])
print(list_1["neighbourhood_group"].isna().all())

#%%
# all columns that are not contained in smaller dataframe
additional_cols = list_2[list_2.columns[~list_2.columns.isin(list_1.columns)]]

# keep all information from listings.csv and join additional information from listings.csv.gz
listings_df = list_1.join(additional_cols)

#%%
calendar_df = pd.read_csv(
    "../data-raw/calendar.csv.gz", parse_dates=["date"], index_col="listing_id"
)

#%%
# Keep only 4 out of 7 Data Frames:
# - listings.csv.gz
# - calendar.csv.gz
# - reviews.csv.gz
# - neighbourhoods.geojson

#%%
# all three dataframes are connected by id columns
print(listings_df.index.nunique())
print(calendar_df.index.nunique())
print(reviews_df.index.nunique())

#%%
# listings_df strict superset of unique observations
print(calendar_df.index.isin(listings_df.index).all())
print(reviews_df.index.isin(listings_df.index).all())

# calendar_df not a superset of reviews_df observations
print(reviews_df.index.isin(calendar_df.index).all())

#%%
# SECTION: Convert Data Types

reviews_df = reviews_df.convert_dtypes()
neighbourhoods_df = neighbourhoods_df.convert_dtypes()

calendar_df = calendar_df.convert_dtypes().assign(
    available=calendar_df["available"].astype("category"),
    # price column
    price=calendar_df["price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype("float"),
    # adjusted_price column
    adjusted_price=calendar_df["adjusted_price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype("float"),
)

listings_df = (
    listings_df.assign(
        first_review=pd.to_datetime(listings_df["first_review"]),
        last_review=pd.to_datetime(listings_df["last_review"]),
        price=listings_df["price"].astype("float"),
        host_is_superhost=listings_df["host_is_superhost"].astype("category"),
        # bathrooms_text column
        bathrooms_text=listings_df["bathrooms_text"].replace(
            {"Half-bath": "0.5 baths", "Shared half-bath": "0.5 shared"}
        ),
        # number_bathrooms column
        number_bathrooms=lambda x: x["bathrooms_text"]
        .str.split(expand=True)
        .iloc[:, 0]
        .apply(pd.to_numeric),
        # shared_bathrooms column
        shared_bathrooms=lambda x: x["bathrooms_text"].str.contains(
            "shared", case=False, regex=False
        ),
        # host_acceptance_rate column
        host_acceptance_rate=listings_df["host_acceptance_rate"]
        .str.replace("%", "")
        .astype("float")
        / 100,
    )
    .rename({"bedrooms": "number_bedrooms"})
    .convert_dtypes()
)

#%%
# Write clean Datasets to file
listings_df.to_pickle(path="listings.pkl")
reviews_df.to_pickle(path="reviews.pkl")
calendar_df.to_pickle(path="calendar.pkl")
neighbourhoods_df.to_pickle(path="neighbourhoods.pkl")

#%%
