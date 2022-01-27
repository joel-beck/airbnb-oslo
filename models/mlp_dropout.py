#%%
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from pytorch_helpers import MLP, run_regression
from sklearn_helpers import ResultContainer, get_column_transformer

sns.set_theme(style="whitegrid")
pd.set_option("precision", 3)

#%%
X_train_val = pd.read_pickle("../data-clean/X_train_val.pkl")
y_train_val = pd.read_pickle("../data-clean/y_train_val.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
def get_data(
    X_train_val: pd.DataFrame, y_train_val: pd.DataFrame
) -> tuple[TensorDataset, TensorDataset]:
    # currently 59 transformed columns
    column_transformer = get_column_transformer()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=123, shuffle=True
    )

    X_train_tensor = torch.tensor(
        column_transformer.fit_transform(X_train, y_train).astype(np.float32)
    )
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
    trainset = TensorDataset(X_train_tensor, y_train_tensor)

    X_val_tensor = torch.tensor(column_transformer.transform(X_val).astype(np.float32))
    y_val_tensor = torch.tensor(y_val.values.astype(np.float32))
    valset = TensorDataset(X_val_tensor, y_val_tensor)

    return trainset, valset, column_transformer


#%%
# BOOKMARK: Hyperparameters
hidden_features_list = [64, 128, 256, 128, 64, 8]

batch_size = 128
num_epochs = 200
use_skip_connections = True
lr = 0.01
log_y = False

dropout_prob_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#%%
neural_network_metrics = []

for dropout_prob in dropout_prob_list:
    trainset, valset, column_transformer = get_data(X_train_val, y_train_val)
    in_features = len(trainset[0][0])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    model = MLP(
        in_features, hidden_features_list, dropout_prob, use_skip_connections
    ).to(device)

    loss_function = nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=lr)

    result_container = ResultContainer(
        model_names=["NeuralNetwork"], feature_selector=[None]
    )
    # append actual number of observed features
    result_container.num_features.append(in_features)
    result_container.selected_features.append(
        [
            feature.split("__")[1]
            for feature in column_transformer.get_feature_names_out()
        ]
    )
    result_container.hyperparam_keys.append(
        ["batch_size", "num_epochs", "dropout_probability"]
    )
    result_container.hyperparam_values.append([batch_size, num_epochs, dropout_prob])

    print(f"Dropout Probability: {dropout_prob}\n")

    metrics, result_container = run_regression(
        model,
        optimizer,
        loss_function,
        device,
        num_epochs,
        trainloader,
        valloader,
        result_container,
        log_y=log_y,
        verbose=True,
        save_best=True,
    )

    metrics.plot()

    current_metrics_df = pd.DataFrame(
        data={
            "dropout_probability": dropout_prob,
            "epochs": range(1, num_epochs + 1),
            "train_mses": metrics.train_mses,
            "val_mses": metrics.val_mses,
            "train_maes": metrics.train_maes,
            "val_maes": metrics.val_maes,
            "train_r2s": metrics.train_r2s,
            "val_r2s": metrics.val_r2s,
        }
    )

    neural_network_metrics.append(current_metrics_df)

metrics_df = pd.concat(neural_network_metrics)


#%%
plot_df = pd.melt(
    metrics_df.rename(
        columns={
            "dropout_probability": "Dropout Probability",
            "epochs": "Epoch",
            "train_maes": "Training MAE",
            "val_maes": "Validation MAE",
            "train_r2s": "Training R2",
            "val_r2s": "Validation R2",
        }
    ),
    id_vars=["Dropout Probability", "Epoch"],
    value_vars=["Training MAE", "Validation MAE", "Training R2", "Validation R2"],
)

g = sns.relplot(
    data=plot_df,
    kind="line",
    x="Epoch",
    y="value",
    hue="Dropout Probability",
    col="variable",
    col_wrap=2,
    legend="full",
    facet_kws=dict(sharey=False),
)

g.set_titles("{col_name}").set_ylabels("")

g.fig.suptitle(
    "Training and Validation Performance for different Dropout Probabilities"
)
g.fig.subplots_adjust(top=0.9)

g.fig.savefig("../term-paper/images/dropout_performance.png")
