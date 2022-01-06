#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
pd.set_option("precision", 3)

#%%
# SUBSECTION: Collect Results
k_best_results = pd.read_pickle("k_best_results.pkl")
rfe_results = pd.read_pickle("rfe_results.pkl")
vt_results = pd.read_pickle("vt_results.pkl")
pca_results = pd.read_pickle("pca_results.pkl")
full_features_results = pd.read_pickle("full_features_results.pkl")

#%%
complete_results = pd.concat(
    [k_best_results, rfe_results, vt_results, pca_results, full_features_results]
).sort_values("mae_val")

complete_results.to_pickle("complete_results.pkl")

#%%
# SUBSECTION: Analyze Results
complete_results

#%%
complete_results.sort_values("r2_val", ascending=False)

#%%
plot_data = (
    complete_results.fillna({"feature_selector": "None"})
    .astype({"feature_selector": "category"})[
        ["mae_val", "r2_val", "num_features", "feature_selector"]
    ]
    .loc[lambda x: x.index != "Mean Prediction"]
)

g = sns.relplot(
    data=plot_data,
    x="mae_val",
    y=plot_data.index,
    hue="num_features",
    col="feature_selector",
    col_wrap=3,
    s=70,
).set(xlabel="", ylabel="", xlim=(500, 800))

g.fig.suptitle("Mean Average Error")
g.fig.subplots_adjust(top=0.9)

#%%
plot_data = plot_data.sort_values("r2_val", ascending=False)

g = sns.relplot(
    data=plot_data,
    x="r2_val",
    y=plot_data.index,
    hue="num_features",
    col="feature_selector",
    col_wrap=3,
    s=70,
).set(xlabel="", ylabel="", xlim=(0, 0.3))

g.fig.suptitle("R^2")
g.fig.subplots_adjust(top=0.9)

#%%
