import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#%%

listings_df = pd.read_pickle("data-clean/listings.pkl")

#%%

# create subsets with first variables to analyize: just take listings_subset first

listings_subset = pd.read_pickle("data-clean/listings_subset.pkl")
listings_subset.head()


#%%

def reg_summary(reg, X, y):
    # The predictions
    y_pred = reg.predict(X)
    # The coefficients
    print(f"Coefficients: \n {reg.coef_} \n Intercept: \n {reg.intercept_}")
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y, y_pred))

    # Plot outputs
    #plt.scatter(X, y, color="black")
    #plt.plot(X, y_pred, color="blue", linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #plt.show()


#%%
listings_subset.isna().sum()
listings_subset = listings_subset.drop(columns=['reviews_per_month', 'host_acceptance_rate']).dropna()

# create dummy variables for categorical variables:

categorical_cols = [
    "neighbourhood",
    "room_type",
    "host_is_superhost",
    "shared_bathrooms",
]


listings_processed = (
    pd.get_dummies(listings_subset, columns=categorical_cols, drop_first=True)
    .loc[listings_df["price"] > 0]
    .dropna()
)


# standardize numerical values

def standardize(df, numeric_cols, train_indices):
    df = df.copy()
    mean_vec = df[numeric_cols].iloc[train_indices].mean()
    std_vec = df[numeric_cols].iloc[train_indices].std()
    df[numeric_cols] = (df[numeric_cols] - mean_vec) / std_vec
    return df

numeric_cols = [
    'price',
    "minimum_nights",
    "number_of_reviews",
    #"reviews_per_month",
    "availability_365",
    #"host_acceptance_rate",
    "number_bathrooms",
    "bedrooms",
    "review_scores_rating",
]

listings_processed = standardize(listings_processed, numeric_cols, train_indices=range(len(listings_processed)))

#%%

X = listings_processed.drop(columns='price')
y = listings_processed['price']



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = LinearRegression().fit(X, y)

reg_summary(reg, X, y)


#%%

plt.scatter(X['number_of_reviews'], y, color="black")
plt.show()


#%%

sns.pairplot(listings_subset[numeric_cols])
plt.show()