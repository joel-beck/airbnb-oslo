# %%
import pandas as pd
import seaborn as sns

# %%
# import clean dataset
listings_df = pd.read_pickle("data-clean/listings.pkl")

# %% [markdown]
# # Exploratory Data Analysis of listings Data Frame

# %%
# SECTION: Exploratory Data Analysis
cols = [
    "price",
    "neighbourhood",
    "room_type",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
]
listings_red = listings_df[cols]


# %%
price_by_nbhood = (
    listings_red.groupby("neighbourhood")
    .agg({"price": ["min", "mean", "max"]})
    .droplevel(level=0, axis="columns")
    .sort_values(by="mean", ascending=False)
)
price_by_nbhood

# %%
g = sns.displot(
    data=listings_red.loc[listings_red["price"] > 0],
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

# %%
(
    sns.catplot(
        data=price_by_nbhood.reset_index(), kind="bar", x="mean", y="neighbourhood"
    )
    .set_axis_labels(x_var="Mean Price", y_var="")
    .set(title="Mean Price by Neighbourhood")
)

# %%
price_by_roomtype = (
    listings_red.groupby("room_type")
    .agg({"price": ["min", "mean", "max"]})
    .droplevel(level=0, axis="columns")
    .sort_values(by="mean", ascending=False)
)

# %%
(
    sns.catplot(
        data=price_by_roomtype.reset_index(), kind="bar", x="mean", y="room_type"
    )
    .set_axis_labels(x_var="Mean Price", y_var="")
    .set(title="Mean Price by Room Type")
)

# %%
sns.relplot(
    kind="scatter",
    data=listings_red.loc[listings_red["price"] > 0],
    x="number_of_reviews",
    y="price",
).set(yscale="log", title="Price vs. # Reviews")

# %%
# english review plus german translation
longest_review = (
    reviews_df["comments"].loc[lambda x: x.str.len() == x.str.len().max()].iloc[0]
)

longest_review

# %%
