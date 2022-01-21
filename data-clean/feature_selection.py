#%%
# from standard library, to convert quoted lists in amenities column to python list
import ast

# pip install gender-guesser, to add predicted gender of hosts based on their name
import gender_guesser.detector as gender
import pandas as pd
from sklearn.model_selection import train_test_split

#%%
listings_df = pd.read_pickle("listings.pkl")
reviews_features = pd.read_pickle("reviews_features.pkl")
front_page_urls = pd.read_pickle("front_page_urls.pkl")

#%%
# SUBSECTION: Add Predicted Host Gender, Number of listed Amenities and Number of Front Page URLs
d = gender.Detector()

listings_df = listings_df.assign(
    host_gender=listings_df["host_name"]
    .apply(d.get_gender)
    .replace({"mostly_male": "male", "mostly_female": "female", "andy": "unknown"}),
    number_amenities=listings_df["amenities"].apply(lambda x: len(ast.literal_eval(x))),
    number_front_page_pictures=front_page_urls.groupby(front_page_urls.index).count(),
)

#%%
# SUBSECTION: Add processed property type
# all property types with N <= 10 sum up in value property_type : "Others"
props = list(listings_df["property_type"].value_counts().loc[lambda x: x <= 10].index)
listings_df["property_type"] = listings_df["property_type"].replace(props, "Others")

#%%
# SUBSECTION: Add Mean Price Prediction for each Apartment from CNN
cnn_predictions = pd.read_pickle("cnn_predictions.pkl")
cnn_pretrained_predictions = pd.read_pickle("cnn_pretrained_predictions.pkl")
 
listings_df = listings_df.assign(
    cnn_predictions=cnn_predictions.groupby(cnn_predictions.index).mean(),
    cnn_pretrained_predictions=cnn_pretrained_predictions.groupby(
        cnn_pretrained_predictions.index
    ).mean(),
)

#%%
# SUBSECTION: Create Extended Dataset with ALL Variables
# These columns cannot be transformed directly into categorical or numeric variables
cols_to_exclude = [
    "amenities",
    "bathrooms_text",  # identical information in number_bathrooms + shared_bathrooms
    "calendar_last_scraped",
    "description",
    "host_id",
    "host_location",
    "host_name",
    "host_picture_url",
    "host_since",
    "host_thumbnail_url",
    "host_url",
    "host_verifications",
    "last_scraped",
    "latitude",
    "listing_url",
    "longitude",
    "name",
    "neighbourhood_cleansed",  # duplicate of neighbourhood
    "picture_url",
    "scrape_id",
]

listings_extended = (
    listings_df.join(reviews_features)
    .drop(columns=cols_to_exclude)
    .loc[lambda x: (x["price"] > 0) & (x["price"] < 80000)]
)

# Drop all variables with more than MAX_MISSING missing observations
MAX_MISSING = 500

incomplete_cols = listings_extended.isna().sum().loc[lambda x: x > MAX_MISSING].index
listings_extended = listings_extended.drop(columns=incomplete_cols).dropna()

listings_extended.to_pickle("listings_extended.pkl")

#%%
# NOTE: Criteria to INCLUDE variables
# - makes theoretical / intuitive sense
# - indicates correlation with price in marginal barplot/scatterplot
# - contains few missing values


# NOTE: Reasons for EXCLUDING specific variables
# host_acceptance_rate: 743 missing values

# host_has_profile_pic: almost no variation (3293 true and 31 false values) and still no marginal correlation with price in barplot
# sns.barplot(data=listings_df, x = "host_has_profile_pic", y = "price")
# listings_df.groupby("host_has_profile_pic").agg(
#     count=("price", "count"), mean_price=("price", "mean")
# )

# host_response_rate: 934 missing values

#%%
# SUBSECTION: Choose Subset with most important columns and drop invalid/missing values

listings_cols = [
    "accomodates",  # added because of overall feature selection
    "availability_30",  # added because of overall feature selection
    "bedrooms",
    # "cnn_predictions",
    "cnn_pretrained_predictions",
    "host_gender",
    "host_identity_verified",
    "host_is_superhost",
    "neighbourhood",
    "minimum_nights_avg_ntm",  # added because of overall feature selection
    "number_amenities",
    "number_bathrooms",
    "number_front_page_pictures",
    "number_of_reviews",
    "price",
    "property_type",  # added because of overall feature selection
    "review_scores_rating",
    "reviews_per_month",
    "room_type",
    "shared_bathrooms",
]

reviews_cols = [
    "frac_norwegian",
    "median_review_length",
    "number_languages",
]

# add numeric features from reviews dataframe to listings_subset,
# join() merges by index
listings_subset = (
    listings_df[listings_cols]
    .join(reviews_features[reviews_cols])
    .loc[(listings_df["price"] > 0) & (listings_df["price"] < 80000)]
    .dropna()
)

listings_subset.to_pickle("listings_subset.pkl")

#%%
# SUBSECTION: Split in Dataset for Model Training and separate Dataset for Evaluation
X = listings_subset.drop(columns="price")
y = listings_subset["price"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, shuffle=True
)

X_train_val.to_pickle("X_train_val.pkl")
X_test.to_pickle("X_test.pkl")
y_train_val.to_pickle("y_train_val.pkl")
y_test.to_pickle("y_test.pkl")

#%%
