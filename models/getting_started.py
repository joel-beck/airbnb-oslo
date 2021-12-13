import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

os.chdir("/Users/marei/airbnb-oslo")

#%%

listings_df = pd.read_pickle("data-clean/listings.pkl")

#%%

# create subsets with first variables to analyize: just take listings_subset first

listings_subset = pd.read_pickle("data-clean/listings_subset.pkl")
listings_subset.head()

#%%

listings_reg = listings_subset[["bedrooms", "price"]].dropna()

x_train = np.array(listings_reg["bedrooms"], dtype=float).reshape(-1, 1)
y_train = np.array(listings_reg["price"])

reg = LinearRegression().fit(x_train, y_train)

reg.score(x_train, y_train)
reg.coef_
reg.intercept_
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

# create dummy variables for categorical values

rooms = pd.get_dummies(listings_subset['room_type'], prefix="room")
superhost = listings_subset["host_is_superhost"].map(dict(t=1, f=0))
shared_bathrooms = listings_subset["shared_bathrooms"].astype(int)

listings_subset = listings_subset.drop(columns=['room_type', 'host_is_superhost', 'shared_bathrooms'])

# stack data frames together

listings_subset = pd.concat([listings_subset, rooms, superhost, shared_bathrooms], axis=1)

#%%

X = listings_subset.drop(columns=['price', 'neighbourhood'])
y = listings_subset['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = LinearRegression().fit(X_train, y_train)

reg_summary(reg, X_train, y_train)


#%%

