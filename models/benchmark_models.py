import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#%%

listings_subset = pd.read_pickle("data-clean/listings_subset.pkl")
listings_subset.head()

listings_processed = pd.read_pickle("data-clean/listings_processed.pkl")
listings_processed.head()

listings_processed.columns

#%%

categorical_cols = [
    "neighbourhood",
    "room_type",
    "host_is_superhost",
    "shared_bathrooms",
]


# function to standardize numerical columns

def standardize(df, numeric_cols, train_indices):
    df = df.copy()
    mean_vec = df[numeric_cols].iloc[train_indices].mean()
    std_vec = df[numeric_cols].iloc[train_indices].std()
    df[numeric_cols] = (df[numeric_cols] - mean_vec) / std_vec
    return df

# select train data

rng = np.random.default_rng(seed=123)
train_frac = 0.8

train_indices = rng.choice(
    range(len(listings_processed)),
    size=int(train_frac * len(listings_processed)),
    replace=False,
)

val_indices = [
    index for index in range(0, len(listings_processed)) if index not in train_indices
]

numeric_cols = [col for col in listings_subset.columns if col not in categorical_cols]

#%%

listings_standardized = standardize(listings_processed, numeric_cols, train_indices)

# split in training and validation set
listings_train = listings_standardized.iloc[train_indices]
listings_val = listings_standardized.iloc[val_indices]

X_train = listings_train.drop(columns='price')
y_train = listings_train['price']

X_val = listings_val.drop(columns='price')
y_val = listings_val['price']

#%%

### 1. Multiple linear regression ###

reg = LinearRegression().fit(X_train, y_train)

def reg_summary(reg, X_test, y_test):
    # The predictions
    y_pred = reg.predict(X_test)
    # The coefficients
    print(f"Intercept: \n {reg.intercept_} \n Coefficients: \n {reg.coef_}")
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


reg_summary(reg, X_val, y_val)


params = np.append(reg.intercept_,reg.coef_)
predictions = reg.predict(X_train)

X = pd.DataFrame(X_train)
X.insert(0, "Constant", np.ones(len(X)))
MSE = (sum((y_train-predictions)**2))/(len(X)-len(X.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
# newX = np.append(np.ones((len(X),1)), X, axis=1)
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
X = np.array(X, dtype=float)
var_b = MSE*(np.linalg.inv(X.T @ X).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i), (len(X)-len(X[0])))) for i in ts_b]

sd_b = np.round(sd_b, 3)
ts_b = np.round(ts_b, 3)
p_values = np.round(p_values, 3)
params = np.round(params, 4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilities"] = [params, sd_b, ts_b, p_values]
print(myDF3)


#%%

### 2. Random Forest ###


#%%

### XGBoost ###