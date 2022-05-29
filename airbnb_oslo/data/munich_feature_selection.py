import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", 100)


def main():
    munich_listings = pd.read_pickle("../../data/munich/munich_listings.pkl")
    munich_reviews_features = pd.read_pickle(
        "../../data/munich/munich_reviews_features.pkl"
    )

    # SUBSECTION: Select Listings Features for Modeling
    subset_cols = [
        "accommodates",
        "availability_30",
        "availability_60",
        "availability_90",
        "availability_365",
        "beds",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
        "calculated_host_listings_count",
        "has_availability",
        # "host_acceptance_rate", # too many missing values
        "host_has_profile_pic",
        "host_identity_verified",
        "host_is_superhost",
        "host_listings_count",
        # "host_response_rate",  # too many missing values
        # "host_response_time",  # too many missing values
        "host_total_listings_count",
        "instant_bookable",
        "latitude",
        "longitude",
        "maximum_nights",
        "minimum_nights",
        "neighbourhood_cleansed",
        "number_amenities",
        "number_bathrooms",
        "number_bedrooms",
        "number_of_reviews",
        "price",
        "property_type",
        "review_scores_accuracy",
        "review_scores_checkin",
        "review_scores_cleanliness",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_rating",
        "review_scores_value",
        "reviews_per_month",
        "room_type",
        "shared_bathrooms",
    ]
    munich_listings_subset = munich_listings[subset_cols]

    reviews_cols = [
        "frac_negative",
        "frac_german",
        "median_review_length",
        "number_languages",
    ]
    reviews_subset = munich_reviews_features[reviews_cols]

    # SUBSECTION: Combine Listings and Reviews Features for Munich
    munich_listings_reviews = munich_listings_subset.join(reviews_subset).dropna()
    munich_listings_reviews.to_pickle("../../data/munich/munich_listings_subset.pkl")

    # SUBSECTION: Split in Dataset for Model Training and separate Dataset for
    # Evaluation
    X = munich_listings_reviews.drop(columns="price")
    y = munich_listings_reviews["price"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, shuffle=True
    )

    X_train_val.to_pickle("../../data/munich/munich_X_train_val.pkl")
    X_test.to_pickle("../../data/munich/munich_X_test.pkl")
    y_train_val.to_pickle("../../data/munich/munich_y_train_val.pkl")
    y_test.to_pickle("../../data/munich/munich_y_test.pkl")


if __name__ == "__main__":
    main()
