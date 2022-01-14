import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


#%%

listings_df = pd.read_pickle("../data-clean/listings.pkl")
#%%

amenities = np.array(listings_df['amenities'])


def list_amenities(X):
    amenities_list = np.array([])
    for i in range(len(X)):
        entry = X[i]
        entry = entry[2:-2]
        entry = np.array(entry.split('", "'))

        amenities_list = np.concatenate((amenities_list, entry), axis=None)

    amenities_list = np.unique(amenities_list)

    return amenities_list

amenities_list = list_amenities(amenities)



#%%

