#%%
import pandas as pd
import seaborn as sns
import ast

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 100)

#%%
# superset of 'munich_listings.csv' file => only consider larger dataset
munich_listings = pd.read_csv("munich_listings.csv.gz", index_col="id")
reviews_df = pd.read_csv(
    "munich_reviews.csv.gz", parse_dates=["date"], index_col="listing_id"
)

munich_listings = pd.DataFrame(munich_listings)

#%%
munich_listings_clean = munich_listings.assign(
    price=munich_listings["price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype("float"),
    first_review=pd.to_datetime(munich_listings["first_review"]),
    last_review=pd.to_datetime(munich_listings["last_review"]),
    # bathrooms_text column
    bathrooms_text=munich_listings["bathrooms_text"].replace(
        {
            "Half-bath": "0.5 baths",
            "Shared half-bath": "0.5 shared",
            "Private half-bath": "1 baths",
        }
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
    host_acceptance_rate=munich_listings["host_acceptance_rate"]
    .str.replace("%", "")
    .astype("float")
    / 100,
    host_response_rate=munich_listings["host_response_rate"]
    .str.replace("%", "")
    .astype("float")
    / 100,
    host_response_time=munich_listings["host_response_time"].astype("category"),
    host_is_superhost=munich_listings["host_is_superhost"].astype("category"),
    host_has_profile_pic=munich_listings["host_has_profile_pic"].astype("category"),
    host_identity_verified=munich_listings["host_identity_verified"].astype("category"),
    instant_bookable=munich_listings["instant_bookable"].astype("category"),
    number_amenities=munich_listings["amenities"].apply(
        lambda x: len(ast.literal_eval(x))
    ),
).rename(columns={"bedrooms": "number_bedrooms"})

#%%
munich_listings_clean = munich_listings_clean.loc[
    (munich_listings_clean["price"] > 0) & (munich_listings_clean["price"] < 9999)
]

min_obs_property_type = 10
rare_categories = (
    munich_listings_clean["property_type"]
    .value_counts()
    .loc[lambda x: x < min_obs_property_type]
    .index
)

munich_listings_clean["property_type"] = (
    munich_listings_clean["property_type"]
    .replace(dict.fromkeys(rare_categories, "Other"))
    .astype("category")
)

#%%
munich_listings_clean.convert_dtypes().to_pickle("munich_listings.pkl")

#%%
