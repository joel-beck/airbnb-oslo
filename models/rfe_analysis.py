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

#%%
complete_rfe_results

#%%
plot_data = complete_rfe_results.astype({"num_features": "category"})

# sharex="col", sharey="row" is amazing :)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex="col", sharey=True)
ax1, ax2, ax3, ax4 = axes.flat

sns.scatterplot(
    data=plot_data,
    x="mae_val",
    y=plot_data.index,
    hue="num_features",
    markers=["s", "o"],
    s=70,
    ax=ax1,
).set(title="MAE Validation", xlabel="")

sns.scatterplot(
    data=plot_data,
    x="mae_train",
    y=plot_data.index,
    hue="num_features",
    markers=["s", "o"],
    s=70,
    ax=ax3,
    legend=False,
).set(title="MAE Training", xlabel="")

sns.scatterplot(
    data=plot_data,
    x="r2_val",
    y=plot_data.index,
    hue="num_features",
    markers=["s", "o"],
    s=70,
    ax=ax2,
    legend=False,
).set(title=r"$R^2$ Validation", xlabel="")

sns.scatterplot(
    data=plot_data,
    x="r2_train",
    y=plot_data.index,
    hue="num_features",
    markers=["s", "o"],
    s=70,
    ax=ax4,
    legend=False,
).set(title=r"$R^2$ Training", xlabel="")

fig.suptitle(
    "Model Performances for different feature sets\n"
    "Neural Network fitted with prices on original scale, all other Models fitted with prices on logarithmic scale"
)
fig.subplots_adjust(right=0.8, top=0.9)

sns.move_legend(obj=ax1, loc="center", bbox_to_anchor=(2.5, -0.1), frameon=False)

plt.show()
