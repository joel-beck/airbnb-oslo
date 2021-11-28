import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Subset, random_split


def generate_train_val_data_split(full_dataset, split_seed=123, val_frac=0.2):
    num_val_samples = np.ceil(val_frac * len(full_dataset)).astype(int)
    num_train_samples = len(full_dataset) - num_val_samples
    trainset, valset = random_split(
        dataset=full_dataset,
        lengths=(num_train_samples, num_val_samples),
        generator=torch.Generator().manual_seed(split_seed),
    )
    return trainset, valset


def generate_subsets(trainset, valset, subset_size):
    train_indices = torch.randint(0, len(trainset) + 1, size=(subset_size,))
    trainset = Subset(dataset=trainset, indices=train_indices)

    val_indices = torch.randint(0, len(valset) + 1, size=(subset_size,))
    valset = Subset(dataset=valset, indices=val_indices)

    return trainset, valset


def print_param_shapes(model, col_widths=(25, 8)):
    for name, param in model.named_parameters():
        print(
            f"Name: {name:<{col_widths[0]}} | # Params: {param.numel():<{col_widths[1]}} | Shape: {list(param.shape)}"
        )
    print("\nTotal number of parameters:", sum(p.numel() for p in model.parameters()))


def _print_shape(input, layer=None, col_width=25):
    if layer is None:
        print(f"{f'Input shape:':<{col_width}} {list(input.shape)}")
    else:
        print(f"{f'{layer.__class__.__name__} output shape:':<25} {list(input.shape)}")


def print_data_shapes(model, device, input_shape, exclude=nn.Sequential):
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


def train_regression(dataloader, optimizer, model, loss_function, device):
    epoch_loss, epoch_total = 0.0, 0.0

    for x, y in dataloader:

        x = x.to(device=device)
        y = y.to(device=device)

        optimizer.zero_grad()
        model.train()

        y_pred = model(x).squeeze()
        batch_size = len(y)
        epoch_total += batch_size

        # Mean Loss per sample
        loss = loss_function(y_pred, y)
        # Loss per minibatch
        epoch_loss += loss * batch_size

        loss.backward()
        optimizer.step()

    return epoch_loss.detach().to(device="cpu").numpy() / epoch_total


def validate_regression(dataloader, model, loss_function, device):
    epoch_loss, epoch_total = 0.0, 0.0

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:

            x = x.to(device=device)
            y = y.to(device=device)

            y_pred = model(x).squeeze()
            batch_size = len(y)
            epoch_total += batch_size

            # Mean Loss per sample
            loss = loss_function(y_pred, y)
            # Loss per minibatch
            epoch_loss += loss * batch_size

    return epoch_loss.detach().to(device="cpu").numpy() / epoch_total


def run_regression(
    model,
    optimizer,
    loss_function,
    device,
    num_epochs,
    train_dataloader,
    val_dataloader,
    save_best=False,
    save_path=None,
    verbose=False,
):
    start_time = time.time()
    train_losses, val_losses = [], []

    if save_best:
        best_loss = np.inf

    for epoch in range(1, num_epochs + 1):

        epoch_train_loss = train_regression(
            dataloader=train_dataloader,
            optimizer=optimizer,
            model=model,
            loss_function=loss_function,
            device=device,
        )
        epoch_val_loss = validate_regression(
            dataloader=val_dataloader,
            model=model,
            loss_function=loss_function,
            device=device,
        )

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if save_best:
            if epoch_val_loss < best_loss:
                best_loss_epoch = epoch
                best_loss = epoch_val_loss

                if save_path is not None:
                    torch.save(model.state_dict(), save_path)

        if verbose:
            if epoch % int(num_epochs / 5) == 0:
                print(f"Epoch: {epoch} / {num_epochs}\n{'-' * 50}")
                print(
                    f"Mean Loss Training: {epoch_train_loss:.5f} | Mean Loss Validation: {epoch_val_loss:.5f}\n"
                )

    time_elapsed = np.round(time.time() - start_time, 0).astype(int)
    print(f"Finished training after {time_elapsed} seconds.")

    if save_best:
        print(f"\nBest Mean Loss Validation: {best_loss:.3f} (Epoch {best_loss_epoch})")

    return train_losses, val_losses


def plot_regression(train_losses, val_losses):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Training", marker="o")
    ax.plot(epochs, val_losses, label="Validation", marker="o")
    ax.set(
        title="Loss",
        xlabel="Epoch",
        ylabel="",
    )
    ax.legend()

    sns.despine()
    plt.show()


def accuracy(correct, total):
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
