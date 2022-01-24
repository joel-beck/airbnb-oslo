#%%
import numpy as np
import pandas as pd

# conda/mamba install langdetect
from langdetect import detect

# conda/mamba install tqdm
from tqdm import tqdm

# to use progress bar / progress_apply() instead of apply()
tqdm.pandas()

#%%
reviews = pd.read_pickle("reviews.pkl")

#%%
def detect_language(review):
    """
    Identifies Language of a single Review, returns missing Value when Identification is not possible
    """

    try:
        language = detect(review)
    except:
        language = pd.NA
    return language


#%%
# SUBSECTION: Detect Languages of all Reviews
# takes about 20 minutes on my cpu
language = reviews["comments"].progress_apply(detect_language)

#%%
language.to_pickle(path="review_languages.pkl")

#%%
reviews_features = pd.DataFrame(
    data={"language": language, "review_length": reviews["comments"].str.len()}
).reset_index()

# SUBSECTION: Drop reviews with unrecognized languages (e.g. only one character long) and save Summary Statistics
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

reviews_features.to_pickle(path="reviews_features.pkl")

#%%
# add results of sentiment analysis
# decided to add number of negative reviews

sentiment_analysis = pd.read_pickle("../exploratory/reviews_sentimentA.pkl")
language = pd.read_pickle("review_languages.pkl")

#%%
reviews_features = pd.DataFrame(
    data={"language": language, "review_length": reviews["comments"].str.len()} #, "label": sentiment_analysis["label"]}
).reset_index()

#%%
# SUBSECTION: Drop reviews with unrecognized languages (e.g. only one character long) and save Summary Statistics
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
        # num_neg_reviews=("label", lambda x: (x=="NEGATIVE").sum()) # tried to add number of negative reviews from sentiment analysis
    )
)

reviews_features.to_pickle(path="reviews_features_extended.pkl")
