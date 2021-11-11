import geopandas as gpd
import pandas as pd

# Keep only 4 out of 7 Data Frames:
# - listings.csv.gz
# - calendar.csv.gz
# - reviews.csv.gz
# - neighbourhoods.geojson

# combine two neighbourhood dataframes
nbhood_1 = pd.read_csv("data-raw/neighbourhoods.csv")
nbhood_2 = gpd.read_file("data-raw/neighbourhoods.geojson")

neighbourhoods_df = pd.merge(
    nbhood_1.drop(columns=["neighbourhood_group"]),
    nbhood_2.drop(columns=["neighbourhood_group"]),
    how="left",
)

# reviews.csv redundant => keep only reviews.csv.gz
reviews_df = pd.read_csv(
    "./data-raw/reviews.csv.gz", parse_dates=["date"], index_col="listing_id"
)

list_1 = pd.read_csv("data-raw/listings.csv", index_col="id")
list_2 = pd.read_csv("data-raw/listings.csv.gz", index_col="id")

# all columns that are not contained in smaller dataframe
additional_cols = list_2[list_2.columns[~list_2.columns.isin(list_1.columns)]]

# keep all information from listings.csv and join additional information from listings.csv.gz
listings_df = list_1.join(additional_cols)

calendar_df = pd.read_csv(
    "data-raw/calendar.csv.gz", parse_dates=["date"], index_col="listing_id"
)

# Convert Data Types
reviews_df = reviews_df.convert_dtypes()
neighbourhoods_df = neighbourhoods_df.convert_dtypes()

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

listings_df = listings_df.assign(
    first_review=pd.to_datetime(listings_df["first_review"]),
    last_review=pd.to_datetime(listings_df["last_review"]),
    price=listings_df["price"].astype("float"),
    host_is_superhost=listings_df["host_is_superhost"].astype("category"),
    bathrooms_text=listings_df["bathrooms_text"].replace(
        {"Half-bath": "0.5 baths", "Shared half-bath": "0.5 shared"}
    ),
    number_bathrooms=lambda x: x["bathrooms_text"]
    .str.split(expand=True)
    .iloc[:, 0]
    .apply(pd.to_numeric),
    shared_bathrooms=lambda x: x["bathrooms_text"].str.contains(
        "shared", case=False, regex=False
    ),
    host_acceptance_rate=listings_df["host_acceptance_rate"]
    .str.replace("%", "")
    .astype("float")
    / 100,
).convert_dtypes()


# Write clean Datasets to file
listings_df.to_pickle(path="data-clean/listings.pkl")
reviews_df.to_pickle(path="data-clean/reviews.pkl")
calendar_df.to_pickle(path="data-clean/calendar.pkl")
neighbourhoods_df.to_pickle(path="data-clean/neighbourhoods.pkl")
