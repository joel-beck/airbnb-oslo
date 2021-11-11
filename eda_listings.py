#%% [markdown]
# # Exploratory Data Analysis of listings Data Frame

#%%
import pandas as pd
import seaborn as sns

#%%
# import clean dataset
listings_df = pd.read_pickle("data-clean/listings.pkl")
listings_df = pd.DataFrame(listings_df)

listings_df.columns

#%%
# SECTION: Exploratory Data Analysis

# most important columns to build model
cols = [
    "price",
    "neighbourhood",
    "room_type",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
    "host_acceptance_rate",
    "host_is_superhost",
    "number_bathrooms",
    "shared_bathrooms",
    "bedrooms",
    "review_scores_rating",
]

# exclude observations where price = 0
listings_subset = listings_df[cols].loc[listings_df["price"] > 0]
listings_subset.head()

#%% [markdown]
#  ## Most expensive neighbourhoods

#%%
price_by_nbhood = (
    listings_subset.groupby("neighbourhood")
    .agg({"price": ["min", "mean", "max"]})
    .droplevel(level=0, axis="columns")
    .sort_values(by="mean", ascending=False)
)
price_by_nbhood


#%%
g = sns.displot(
    data=listings_subset.loc[listings_subset["price"] > 0],
    kind="hist",
    x="price",
    hue="neighbourhood",
    col="neighbourhood",
    col_wrap=4,
    log_scale=True,
    facet_kws=dict(sharey=False),
    legend=False,
    height=2,
    aspect=1,
    col_order=price_by_nbhood.index,
).set_titles(col_template="{col_name}")

fig = g.figure
fig.subplots_adjust(top=0.9)
fig.suptitle("Log-Price Distribution")


#%%
(
    sns.catplot(
        data=price_by_nbhood.reset_index(), kind="bar", x="mean", y="neighbourhood"
    )
    .set_axis_labels(x_var="Mean Price", y_var="")
    .set(title="Mean Price by Neighbourhood")
)


#%% [markdown]
#  ## Most expensive room types

#%%
# exclude prices of 0
price_by_roomtype = (
    listings_subset.groupby("room_type")
    .agg({"price": ["min", "mean", "max"]})
    .droplevel(level=0, axis="columns")
    .sort_values(by="mean", ascending=False)
)
price_by_roomtype

#%%
(
    sns.catplot(
        data=price_by_roomtype.reset_index(), kind="bar", x="mean", y="room_type"
    )
    .set_axis_labels(x_var="Mean Price", y_var="")
    .set(title="Mean Price by Room Type")
)


#%% [markdown]
#  ## Are rooms with more reviews more or less expensive?

#%%
sns.relplot(
    kind="scatter",
    data=listings_subset,
    x="number_of_reviews",
    y="price",
).set(yscale="log", title="Price vs. # Reviews")

#%%
