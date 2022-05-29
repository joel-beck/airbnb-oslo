import numpy as np
import pandas as pd
from langdetect import detect
from tqdm import tqdm

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
    reviews = pd.read_pickle("../../data/clean/reviews.pkl")

    # SUBSECTION: Detect Languages of all Reviews
    # takes about 20 minutes on my cpu
    language = reviews["comments"].progress_apply(detect_language)
    language.to_pickle("../../data/clean/review_languages.pkl")

    # SUBSECTION: Drop reviews with unrecognized languages (e.g. only one character
    # long) and add Summary Statistics
    language = pd.read_pickle("../../data/clean/review_languages.pkl")

    reviews_features = pd.DataFrame(
        data={"language": language, "review_length": reviews["comments"].str.len()}
    ).reset_index()

    reviews_features = (
        reviews_features.dropna(subset=["language"])
        .groupby("listing_id")
        .agg(
            number_reviews=("language", lambda x: x.size),
            median_review_length=("review_length", lambda x: np.median(x)),
            number_languages=("language", lambda x: x.nunique()),
            frac_english=("language", lambda x: (x == "en").mean()),
            frac_norwegian=("language", lambda x: (x == "no").mean()),
            frac_missing=("language", lambda x: x.isna().mean()),
            language_list=("language", lambda x: x.unique()),
        )
    )

    # SUBSECTION: Add fraction of negative reviews as new column
    sentiment_analysis = pd.read_pickle("../../data/clean/reviews_sentiment.pkl")

    frac_negative = sentiment_analysis.groupby(sentiment_analysis.index).agg(
        frac_negative=("label", lambda x: (x == "NEGATIVE").mean())
    )
    reviews_features = reviews_features.assign(frac_negative=frac_negative)
    reviews_features.to_pickle("../../data/clean/reviews_features.pkl")


if __name__ == "__main__":
    main()
