import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from sklearn_helpers import ResultContainer


def generate_train_val_data_split(
    full_dataset: Dataset, split_seed: int = 123, val_frac: float = 0.2
) -> tuple[Dataset, Dataset]:
    """
    Splits the entire Dataset used for Model Training into a Training Set and Validation Set.
    The relative Sizes of each Output Component can be specified with a fractional Value between 0 and 1.
    """

    num_val_samples = np.ceil(val_frac * len(full_dataset)).astype(int)
    num_train_samples = len(full_dataset) - num_val_samples
    trainset, valset = random_split(
        dataset=full_dataset,
        lengths=(num_train_samples, num_val_samples),
        generator=torch.Generator().manual_seed(split_seed),
    )
    return trainset, valset


def generate_subsets(
    trainset: Dataset, valset: Dataset, subset_size: int
) -> tuple[Dataset, Dataset]:
    """
    Returns Training and Validation Sets of Reduced Size by randomly sampling observations from the original Training and Validation Sets.
    The Sizes can be specified with an Integer Input Value.

    This procedure is useful at early Stages of Model Construction for confirming a correct Model Setup as well as for Debugging on a CPU.
    """

    train_indices = torch.randint(0, len(trainset) + 1, size=(subset_size,))
    trainset = Subset(dataset=trainset, indices=train_indices)

    val_indices = torch.randint(0, len(valset) + 1, size=(subset_size,))
    valset = Subset(dataset=valset, indices=val_indices)

    return trainset, valset


def init_data_loaders(
    trainset: Dataset, valset: Dataset, testset: Dataset, batch_size: int = 64
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns PyTorch DataLoader Objects for the Training, Validation and Test Set
    """

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, valloader, testloader


def print_param_shapes(model: Any, col_widths: tuple[int, int] = (25, 8)):
    """
    Prints the Shape of the Weight Tensors between all Layers in a Neural Network as well as the total Number of Parameters trained during Model Fitting.
    """

    for name, param in model.named_parameters():
        print(
            f"Name: {name:<{col_widths[0]}} | # Params: {param.numel():<{col_widths[1]}} | Shape: {list(param.shape)}"
        )
    print("\nTotal number of parameters:", sum(p.numel() for p in model.parameters()))


def _print_shape(input: torch.Tensor, layer: Optional[Any] = None, col_width: int = 25):
    """
    Prints the Shape of the Output Tensor for a single Layer in a Neural Network.
    """

    if layer is None:
        print(f"{f'Input shape:':<{col_width}} {list(input.shape)}")
    else:
        print(f"{f'{layer.__class__.__name__} output shape:':<25} {list(input.shape)}")


def print_data_shapes(
    model: Any,
    device: torch.device,
    input_shape: tuple[int, ...],
    exclude: Union[nn.Sequential, list] = nn.Sequential,
):
    """
    Prints the Shape of the Output Tensor for each Layer in a Neural Network.
    This can be particularly helpful for specifiying the correct dimensions after flattening a four-dimensional Tensor in a Convolutional Network.
    """

    x = torch.rand(size=input_shape, dtype=torch.float32).to(device=device)
    _print_shape(x)

    for i, layer in enumerate(model.modules()):
        if i == 0:
            continue

        elif isinstance(layer, exclude):
            print("-" * 20)
            print(f"{layer.__class__.__name__} layer:")

        else:
            x = layer(x)
            _print_shape(x, layer)


class MLP(nn.Module):
    """
    Architecture of Fully-Connected Neural Network for all numeric / metric Input Features
    """

    def __init__(self, in_features, hidden_features_list, dropout_prob):
        super(MLP, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features_list[0], bias=False),
            nn.BatchNorm1d(hidden_features_list[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.hidden_layers = self.hidden_block(
            in_features=hidden_features_list[0],
            out_features_list=hidden_features_list[1:],
            dropout_prob=dropout_prob,
        )

        self.output_layer = nn.Linear(
            in_features=hidden_features_list[-1], out_features=1
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def hidden_block(self, in_features, out_features_list, dropout_prob):
        layers = []
        for out_features in out_features_list:
            layers.append(nn.Linear(in_features, out_features, bias=False))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            in_features = out_features

        return nn.Sequential(*layers)


@dataclass
class NeuralNetMetrics:
    """
    Tracks Performance Metrics for all Epochs during Model Training.
    This is particularly useful to plot Loss Curves of the Mean Squared Error, Mean Absolute Error and R^2 Value on Training and Validation Set after Training is completed.
    """

    train_losses: list[str] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_maes: list[float] = field(default_factory=list)
    val_maes: list[float] = field(default_factory=list)
    train_r2s: list[float] = field(default_factory=list)
    val_r2s: list[float] = field(default_factory=list)

    def append(
        self,
        train_loss: float,
        val_loss: float,
        train_mae: float,
        val_mae: float,
        train_r2: float,
        val_r2: float,
    ):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_maes.append(train_mae)
        self.val_maes.append(val_mae)
        self.train_r2s.append(train_r2)
        self.val_r2s.append(val_r2)

    def plot(self):
        sns.set_theme(style="whitegrid")

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(9, 9))

        epochs = range(1, len(self.train_losses) + 1)

        ax1.plot(epochs, self.train_losses, label="Training")  # , marker="o")
        ax1.plot(epochs, self.val_losses, label="Validation")  # , marker="o")
        ax1.set(
            title="Mean Squared Error",
            xlabel="",
            ylabel="",
        )
        ax1.legend()

        ticks_loc = ax1.get_yticks().tolist()
        ax1.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax1.set_yticklabels(["{:,}".format(int(x)) for x in ticks_loc])

        ax2.plot(epochs, self.train_maes, label="Training")  # , marker="o")
        ax2.plot(epochs, self.val_maes, label="Validation")  # , marker="o")
        ax2.set(
            title="Mean Absolute Error",
            xlabel="",
            ylabel="",
        )

        ax3.plot(epochs, self.train_r2s, label="Training")  # , marker="o")
        ax3.plot(epochs, self.val_r2s, label="Validation")  # , marker="o")
        ax3.set(
            title="R2",
            xlabel="Epoch",
            ylabel="",
        )

        fig.tight_layout()
        sns.move_legend(
            obj=ax1, loc="upper center", bbox_to_anchor=(1.1, -0.7), frameon=False
        )

        sns.despine()
        plt.show()


def init_weights(layer: nn.Module, mean: float = 0, std: float = 1):
    """
    Initializes Model Weights at the Start of Training.
    Avoids negative predicted Prices during the first Epochs to take the Logarithm when training with Log-Prices.
    """

    if isinstance(layer, nn.Linear):
        # avoid negative predicted prices at beginning of training to enable log transformation
        torch.nn.init.normal_(layer.weight, mean=mean, std=std)


def train_regression(
    dataloader: DataLoader,
    optimizer: Union[Adam, SGD],
    model: Any,
    loss_function: nn.MSELoss,
    device: torch.device,
    log_y: bool = False,
) -> tuple[float, float, float]:
    """
    Model Training and Weight Adjustment Step on the Training Set for a single Epoch.
    Returns Mean Squared Error, Mean Absolute Error and R^2 Value for this Epoch on the Training Set.
    """

    # calculate mean squared error, mean_absolute error and r2 with values of all batches to perform comparable computations as with classical models
    y_true_list = []
    y_pred_list = []

    model.train()
    for x, y in dataloader:
        x = x.to(device=device)
        y = y.to(device=device)

        optimizer.zero_grad()
        y_pred = model(x).squeeze()

        # train model / backpropagate loss on log scale
        if log_y:
            y = torch.log(y)
            y_pred = torch.log(y_pred)

        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()

        # collect error metrics on original scale
        if log_y:
            y = torch.exp(y)
            y_pred = torch.exp(y_pred)

        y_true_list.extend(list(y.detach()))
        y_pred_list.extend(list(y_pred.detach()))

    mse = mean_squared_error(y_true_list, y_pred_list)
    mae = mean_absolute_error(y_true_list, y_pred_list)
    r2 = r2_score(y_true_list, y_pred_list)

    return mse, mae, r2


def validate_regression(
    dataloader: DataLoader, model: Any, device: torch.device
) -> tuple[float, float, float]:
    """
    Evaluation Step on the Validation Set for a single Epoch.
    Returns Mean Squared Error, Mean Absolute Error and R^2 Value for this Epoch on the Validation Set.
    """

    y_true_list = []
    y_pred_list = []

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)

            y_pred = model(x).squeeze()

            y_true_list.extend(list(y.detach()))
            y_pred_list.extend(list(y_pred.detach()))

    mse = mean_squared_error(y_true_list, y_pred_list)
    mae = mean_absolute_error(y_true_list, y_pred_list)
    r2 = r2_score(y_true_list, y_pred_list)

    return mse, mae, r2


def print_epoch(
    epoch: int,
    num_epochs: int,
    epoch_train_mse: float,
    epoch_val_mse: float,
    epoch_train_mae: float,
    epoch_val_mae: float,
    epoch_train_r2: float,
    epoch_val_r2: float,
    scheduler: Optional[Any] = None,
):
    """
    Prints Information about the Model Performance in Training and Validation Set of the current Epoch while the Model is trained.
    """

    print(f"Epoch: {epoch} / {num_epochs}\n{'-' * 50}")
    if scheduler is not None:
        print(f"Learning Rate: {scheduler.state_dict()['_last_lr'][0]:.1e}")
    print(
        f"Mean MSE Training: {epoch_train_mse:.3f} | Mean MSE Validation: {epoch_val_mse:.3f}\n"
        f"Mean MAE Training: {epoch_train_mae:.3f} | Mean MAE Validation: {epoch_val_mae:.3f}\n"
        f"Mean R2 Training: {epoch_train_r2:.3f} | Mean R2 Validation: {epoch_val_r2:.3f}\n"
    )


def print_best(
    best_train_mae: float,
    best_train_mae_epoch: int,
    best_val_mae: float,
    best_val_mae_epoch: int,
):
    """
    Prints the lowest Mean Absolute Error of the Training and Validation Set encountered during Model Training after Training is completed.
    """

    print(
        f"\nBest Mean MAE Training: {best_train_mae:.3f} (Epoch {best_train_mae_epoch})"
        f"\nBest Mean MAE Validation: {best_val_mae:.3f} (Epoch {best_val_mae_epoch})"
    )


def run_regression(
    model: Any,
    optimizer: Union[Adam, SGD],
    loss_function: nn.MSELoss,
    device: torch.device,
    num_epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    result_container: Optional[ResultContainer] = None,
    log_y: bool = False,
    scheduler: Optional[Any] = None,
    save_best: bool = False,
    save_path: bool = None,
    verbose: bool = False,
) -> Union[tuple[NeuralNetMetrics, ResultContainer], NeuralNetMetrics]:
    """
    Trains a Neural Network Regression Model for a specified number of Epochs.
    During Training Performance Information is displayed.
    Optionally the State with the lowest Mean Absolute Error can be tracked and saved.

    Returns a NeuralNetMetrics Object for plotting Loss Curves and optionally a ResultContainer Object to collect the Results in a Pandas DataFrame for Comparison with the Classical Statistical Models.
    """

    start_time = time.perf_counter()

    if result_container is not None:
        result_container.log_y.append(log_y)

    metrics = NeuralNetMetrics()

    if save_best:
        # use mean absolute error as metric for early stopping
        best_train_mae = np.inf
        best_val_mae = np.inf

    for epoch in range(1, num_epochs + 1):

        epoch_train_mse, epoch_train_mae, epoch_train_r2 = train_regression(
            dataloader=train_dataloader,
            optimizer=optimizer,
            model=model,
            loss_function=loss_function,
            device=device,
            log_y=log_y,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_val_mse, epoch_val_mae, epoch_val_r2 = validate_regression(
            dataloader=val_dataloader,
            model=model,
            device=device,
        )

        metrics.append(
            epoch_train_mse,
            epoch_val_mse,
            epoch_train_mae,
            epoch_val_mae,
            epoch_train_r2,
            epoch_val_r2,
        )

        if save_best:
            if epoch_train_mae < best_train_mae:
                best_train_mae_epoch = epoch
                best_train_mae = epoch_train_mae

            if epoch_val_mae < best_val_mae:
                best_val_mae_epoch = epoch
                best_val_mae = epoch_val_mae

                # save weights for lowest validation mae
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)

        if verbose:
            if epoch % int(num_epochs / 5) == 0:
                print_epoch(
                    epoch,
                    num_epochs,
                    epoch_train_mse,
                    epoch_val_mse,
                    epoch_train_mae,
                    epoch_val_mae,
                    epoch_train_r2,
                    epoch_val_r2,
                    scheduler,
                )

    time_elapsed = np.round(time.perf_counter() - start_time, 0).astype(int)
    print(f"Finished training after {time_elapsed} seconds.")

    # check twice for save_best to include both cases for result_container is None and result_container is not None
    if save_best:
        print_best(
            best_train_mae, best_train_mae_epoch, best_val_mae, best_val_mae_epoch
        )

    if result_container is None:
        return metrics

    if save_best:
        # if save_best=True save results from epoch with best validation mae (starts at epoch=1)
        result_container.append(
            metrics.train_maes[best_val_mae_epoch - 1],
            metrics.val_maes[best_val_mae_epoch - 1],
            metrics.train_r2s[best_val_mae_epoch - 1],
            metrics.val_r2s[best_val_mae_epoch - 1],
            metrics.train_losses[best_val_mae_epoch - 1],
            metrics.val_losses[best_val_mae_epoch - 1],
        )
    else:
        # if save_best=False save result from last epoch (starts at epoch=1)
        result_container.append(
            metrics.train_maes[epoch - 1],
            metrics.val_maes[epoch - 1],
            metrics.train_r2s[epoch - 1],
            metrics.val_r2s[epoch - 1],
            metrics.train_losses[epoch - 1],
            metrics.val_losses[epoch - 1],
        )

    return metrics, result_container


# NOTE: The Classification Functions below are currently not used, but kept for optionally using some of their components in the corresponding Regression Functions
def accuracy(correct: int, total: int) -> float:
    return np.round(correct / total, decimals=3)


def train_classification(dataloader, optimizer, model, loss_fn, device):

    # option 1: save epoch losses in list and return mean of this list
    # -> creates unnecessary list structure for each epoch
    # -> samples might not have equal weights if len(dataset) % batch_size != 0, can be sensitive to outliers when samples are shuffled (worst case: last batch contains a single sample, which is an outlier with high loss )
    # option 2: save total epoch loss and length of dataset 8needed for accuracy as well) and return mean sample loss (or mean batch loss when multiplied with batch_size)

    epoch_loss, epoch_correct, epoch_total = 0.0, 0.0, 0.0

    for x, y in dataloader:
        x = x.to(device=device)
        y = y.to(device=device)

        optimizer.zero_grad()
        model.train()

        y_pred = model(x)
        class_pred = y_pred.argmax(dim=1)

        batch_correct = (y == class_pred).sum().item()
        batch_size = len(y)

        epoch_correct += batch_correct
        epoch_total += batch_size

        loss = loss_fn(y_pred, y)
        # Cross Entropy Loss calculates mean loss per sample in batch with default: reduction = "mean"
        epoch_loss += loss * batch_size

        loss.backward()
        optimizer.step()

    return (
        epoch_loss.detach().to(device="cpu").numpy() / epoch_total,
        accuracy(epoch_correct, epoch_total),
    )


def validate_classification(dataloader, model, loss_fn, device):

    epoch_loss, epoch_correct, epoch_total = 0.0, 0.0, 0.0

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device=device)
            y = y.to(device=device)

            y_pred = model(x)
            class_pred = y_pred.argmax(dim=1)

            batch_correct = (y == class_pred).sum().item()
            batch_size = len(y)

            epoch_correct += batch_correct
            epoch_total += batch_size

            loss = loss_fn(y_pred, y)
            # Cross Entropy Loss calculates mean loss per sample in batch with default: reduction = "mean"
            epoch_loss += loss * batch_size

    return (
        epoch_loss.detach().to(device="cpu").numpy() / epoch_total,
        accuracy(epoch_correct, epoch_total),
    )


def run_classification(
    model,
    optimizer,
    loss_function,
    device,
    num_epochs,
    train_dataloader,
    val_dataloader,
    scheduler=None,
    save_best=False,
    save_path=None,
    early_stopper=None,
    verbose=False,
):

    start_time = time.perf_counter()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    if save_best:
        best_loss = np.inf
        best_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):

        epoch_train_loss, epoch_train_acc = train_classification(
            dataloader=train_dataloader,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_function,
            device=device,
        )
        epoch_val_loss, epoch_val_acc = validate_classification(
            dataloader=val_dataloader,
            model=model,
            loss_fn=loss_function,
            device=device,
        )

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        if scheduler is not None:
            schedulers = {
                "StepLR": [],
                "ExponentialLR": [],
                "ReduceLROnPlateau": [epoch_val_loss],
            }
            params = schedulers[scheduler.__class__.__name__]
            scheduler.step(*params)

        if save_best:
            if epoch_val_loss < best_loss:
                best_loss_epoch = epoch
                best_loss = epoch_val_loss

            if epoch_val_acc > best_accuracy:
                best_acc_epoch = epoch
                best_accuracy = epoch_val_acc

                if save_path is not None:
                    torch.save(model.state_dict(), save_path)

        if early_stopper is not None:
            early_stopper.update(val_acc=epoch_val_acc, model=model)
            if early_stopper.early_stop:
                break

        if verbose:
            if epoch % int(num_epochs / 5) == 0:
                print(f"Epoch: {epoch} / {num_epochs}\n{'-' * 50}")
                if scheduler is not None:
                    print(f"Learning Rate: {scheduler.state_dict()['_last_lr'][0]:.1e}")
                print(
                    f"Mean Loss Training: {epoch_train_loss:.5f} | Mean Loss Validation: {epoch_val_loss:.5f}\n"
                    f"Training Accuracy: {100 * epoch_train_acc:.1f}% | Validation Accuracy: {100 * epoch_val_acc:.1f}%\n"
                )

    time_elapsed = np.round(time.perf_counter() - start_time, 0).astype(int)
    print(f"Finished training after {time_elapsed} seconds.")

    if save_best:
        print(
            f"\nBest Mean Loss Validation: {best_loss:.3f} (Epoch {best_loss_epoch})\n"
            f"Best Validation Accuracy: {100 * best_accuracy:.1f}% (Epoch {best_acc_epoch})"
        )

    return train_losses, val_losses, train_accs, val_accs


def plot_classification(train_losses, val_losses, train_accs, val_accs):
    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Training", marker="o")
    ax1.plot(epochs, val_losses, label="Validation", marker="o")
    ax1.set(
        title="Loss",
        xlabel="Epoch",
        ylabel="",
    )
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Training", marker="o")
    ax2.plot(epochs, val_accs, label="Validation", marker="o")
    ax2.set(
        title="Accuracy",
        xlabel="Epoch",
        ylabel="",
        ylim=(0, 1),
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    sns.move_legend(
        obj=ax1, loc="upper center", bbox_to_anchor=(1.1, 1.3), ncol=2, frameon=False
    )
    sns.despine()

    plt.show()
