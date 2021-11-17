import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn


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


def print_data_shapes(model, device, input_shape):
    x = torch.rand(size=input_shape, dtype=torch.float32).to(device=device)
    _print_shape(x)

    for i, layer in enumerate(model.modules()):
        if i == 0 or isinstance(layer, nn.Sequential):
            continue

        else:
            x = layer(x)
            _print_shape(x, layer)


def train_epoch(dataloader, optimizer, model, loss_function, device):
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


def validate_epoch(dataloader, model, loss_function, device):
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


def run_training(
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

        epoch_train_loss = train_epoch(
            dataloader=train_dataloader,
            optimizer=optimizer,
            model=model,
            loss_function=loss_function,
            device=device,
        )
        epoch_val_loss = validate_epoch(
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


def plot_results(train_losses, val_losses):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Training")
    ax.plot(epochs, val_losses, label="Validation")
    ax.set(
        title="Loss",
        xlabel="Epoch",
        ylabel="",
    )
    ax.legend()

    sns.despine()
    plt.show()
