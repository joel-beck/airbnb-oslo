#%%
from warnings import simplefilter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

simplefilter(action="ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
pd.set_option("precision", 3)
pd.set_option("display.max_columns", 100)

#%%
# SECTION: Analyze Performance on Validation Set during Training
classical_models_rfe_results = pd.read_pickle(
    "../results-pickle/classical_models_rfe_results.pkl"
)
neural_network_rfe_results = pd.read_pickle(
    "../results-pickle/neural_network_rfe_results.pkl"
)

complete_rfe_results = pd.concat(
    [
        classical_models_rfe_results,
        neural_network_rfe_results,
    ]
).sort_values("mae_val")

complete_rfe_results.to_pickle("../results-pickle/complete_rfe_results.pkl")
