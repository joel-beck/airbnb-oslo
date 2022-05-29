import numpy as np
import pandas as pd
from langdetect import detect
from tqdm import tqdm
from transformers import pipeline

# to use progress bar / progress_apply() instead of apply()
tqdm.pandas()


def detect_language(review):
    """
    Identifies Language of a single Review, returns missing Value when Identification is
    not possible.
    """
    try:
        language = detect(review)
    except Exception:
        language = pd.NA
    return language


def main():
    reviews_df = pd.read_csv(
        "munich_reviews.csv.gz", parse_dates=["date"], index_col="listing_id"
    )

    # SUBSECTION: Detect Languages of all Reviews
    # takes about 20 minutes on my cpu
    language = reviews_df["comments"].progress_apply(detect_language)
    language.to_pickle(path="../../data/munich/munich_review_languages.pkl")

    language = pd.read_pickle("../../data/munich/munich_review_languages.pkl")
    reviews_df["language"] = language

    sentiment_analyizer = pipeline("sentiment-analysis")
    sentiment_df = pd.DataFrame(
        columns=["listing_id", "id", "comment", "label", "score", "language"]
    )

    # decide if comment is positive or negative and save in new data frame
    n = len(reviews_df["comments"])
    for i in range(n):
        comment = reviews_df["comments"].iloc[i]

        try:
            result = sentiment_analyizer(comment)[0]
        except Exception:
            label = "None"
            score = "None"
        else:
            label = result["label"]
            score = result["score"]

        new_row = {
            "listing_id": reviews_df["comments"].index[i],
            "id": reviews_df["id"].iloc[i],
            "comment": comment,
            "label": label,
            "score": score,
            "language": reviews_df["language"].iloc[i],
        }
        sentiment_df = sentiment_df.append(new_row, ignore_index=True)

        if i % 100 == 0:
            print(i)

    sentiment_df = sentiment_df.set_index("listing_id")
    sentiment_df.to_pickle(path="../../data/munich/munich_reviews_sentiment.pkl")
    sentiment_df = pd.read_pickle("../../data/munich/munich_reviews_sentiment.pkl")

    # SUBSECTION: Drop reviews with unrecognized languages (e.g. only one character
    # long) and add Summary Statistics
    language = pd.read_pickle("../../data/munich/munich_review_languages.pkl")

    reviews_features = pd.DataFrame(
        data={
            "language": language,
            "review_length": reviews_df["comments"].str.len(),
            "sentiment": sentiment_df["label"],
        }
    ).reset_index()

    reviews_features = (
        reviews_features.dropna(subset=["language"])
        .groupby("listing_id")
        .agg(
            number_reviews=("language", lambda x: x.size),
            median_review_length=("review_length", lambda x: np.median(x)),
            number_languages=("language", lambda x: x.nunique()),
            frac_english=("language", lambda x: (x == "en").mean()),
            frac_german=("language", lambda x: (x == "de").mean()),
            language_list=("language", lambda x: x.unique()),
            frac_negative=("sentiment", lambda x: (x == "NEGATIVE").mean()),
        )
    )

    reviews_features.to_pickle(path="../../data/munich/munich_reviews_features.pkl")


if __name__ == "__main__":
    main()
