# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn
from torch.utils.data import TensorDataset

import nneuroutil.array_api_extras as xpx
from nneuroutil.data import SlicedDataLoader
from nneuroutil.helpers import TicTocTimer

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger("lib.torchlib")

# {{{ MultilayerPerceptron


MLP_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "logistic": nn.Sigmoid,
}


class MultilayerPerceptron(nn.Module):
    """A simple feed-forward network that returns class logits.

    This implements the essentials of :class:`sklearn.neural_network.MLPClassifier`:
    a stack of fully connected layers with a configurable activation applied to
    every hidden layer and no activation on the output layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: tuple[int, ...],
        n_classes: int,
        *,
        activation: Literal["relu", "tanh", "logistic"] = "relu",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if activation not in MLP_ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation!r}")
        activation_cls = MLP_ACTIVATIONS[activation]

        from itertools import pairwise

        sizes = (in_features, *hidden_layer_sizes, n_classes)
        layers: list[nn.Module] = []
        for i, (n_in, n_out) in enumerate(pairwise(sizes)):
            layers.append(nn.Linear(n_in, n_out, dtype=dtype, device=device))

            # NOTE: do not add activations to the last layer to match sklearn
            if i < len(sizes) - 2:
                layers.append(activation_cls())

        self.model = nn.Sequential(*layers)
        self.in_features = in_features
        self.out_features = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# }}}


# {{{ MultilayerPerceptronClassifier


class MultilayerPerceptronClassifier:
    """Estimator obtained from :func:`classify_multilayer_perceptron`."""

    def __init__(
        self,
        model: MultilayerPerceptron,
        labels: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.model = model
        self.labels = labels
        self.dtype = dtype
        self.device = device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels for given sample(s) *x*.

        :arg x: 2D tensor of samples of shape ``(nsamples, nfeatures)``.
        :returns: predicted class label(s) taken from the original *labels*.
        """
        self.model.eval()

        with torch.no_grad():
            logits = self.model(x.to(dtype=self.dtype, device=self.device))
            idx = torch.argmax(logits, dim=1)

        return self.labels[idx]


# }}}


# {{{ classify_multilayer_perceptron


def _l2_penalty(model: nn.Module) -> torch.Tensor:
    """Sum of squared weight matrices, excluding biases (as in scikit-learn)."""
    penalty: torch.Tensor | None = None
    for p in model.parameters():
        if p.dim() < 2:
            continue

        term = (p * p).sum()
        penalty = term if penalty is None else penalty + term

    assert penalty is not None
    return penalty


def classify_multilayer_perceptron(
    features: Sequence[torch.Tensor],
    labels: Sequence[Any],
    *,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    activation: Literal["relu", "tanh", "logistic"] = "relu",
    alpha: float = 1.0e-4,
    learning_rate_init: float = 1.0e-3,
    batch_size: int = 4096,
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 10,
    tol: float = 1.0e-4,
    rng: xpx.RandomGenerator | None = None,
) -> MultilayerPerceptronClassifier:
    """Train a multi-layer perceptron (MLP) classifier for multi-class data.

    This mirrors the essential options of :class:`sklearn.neural_network.MLPClassifier`:
    the network is trained with Adam on a categorical cross-entropy loss with an
    optional L2 penalty and optional early stopping on a held-out validation set.

    :arg features: sequence of 2D tensors containing sample features for each
        class, each of shape ``(n_samples_i, nfeatures)``.
    :arg labels: sequence of labels corresponding to each class in *features*.
    :arg hidden_layer_sizes: sizes of the hidden layers, e.g. ``(100,)`` for a
        single hidden layer or ``(64, 32)`` for two.
    :arg activation: activation function applied to the hidden layers.
    :arg alpha: L2 penalty (weight decay) coefficient. As in scikit-learn, the
        penalty only applies to the weight matrices (biases are not penalized)
        and is scaled by the batch size, i.e. the objective is
        ``mean_cross_entropy + 0.5 * alpha * ||W||^2 / batch_size``.
    :arg learning_rate_init: initial learning rate for the Adam optimizer.
    :arg batch_size: number of samples per gradient step.
    :arg max_iter: maximum number of epochs.
    :arg early_stopping: if *True*, hold out a fraction of the training data and
        stop when the validation loss stops improving.
    :arg validation_fraction: fraction of the training data held out for early
        stopping when *early_stopping* is *True*.
    :arg n_iter_no_change: number of epochs without improvement before stopping.
    :arg tol: minimum improvement in the validation loss to count as progress.
    :arg random_state: seed used for the train/validation split and batching.
    """

    if len(features) != len(labels):
        raise ValueError(
            f"'features' and 'labels' do not match: {len(features)} and {len(labels)}"
        )

    if len(features) <= 1:
        raise ValueError(
            "cannot perform classification on 'features' with 0 or 1 classes"
        )

    _, nfeatures = features[0].shape
    dtype = features[0].dtype
    device = features[0].device

    if not dtype.is_floating_point:
        raise ValueError(f"'features' must be floating point: {dtype}")

    if any(f.ndim != 2 or f.shape[-1] != nfeatures for f in features[1:]):
        raise ValueError(
            f"'features' do not have the same number of features: '{nfeatures}'"
        )

    if alpha < 0:
        raise ValueError(f"'alpha' must be non-negative: {alpha}")

    if learning_rate_init <= 0:
        raise ValueError(f"'learning_rate_init' must be positive: {learning_rate_init}")

    if batch_size <= 0:
        raise ValueError(f"'batch_size' must be positive: {batch_size}")

    if max_iter <= 0:
        raise ValueError(f"'max_iter' must be positive: {max_iter}")

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            f"'validation_fraction' must be in [0, 1): {validation_fraction}"
        )

    # concatenate the per-class features and build the matching index targets
    x = torch.cat(list(features), dim=0)
    y = torch.cat([
        torch.full((f.shape[0],), i, dtype=torch.long, device=device)
        for i, f in enumerate(features)
    ])
    class_labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    model = MultilayerPerceptron(
        nfeatures,
        hidden_layer_sizes,
        len(features),
        activation=activation,
        dtype=dtype,
        device=device,
    )

    # {{{ split a validation set for early stopping

    x_train, y_train = x, y
    x_validate = y_validate = None
    if early_stopping and validation_fraction > 0.0:
        from nneuroutil.data import random_split

        (x_train, y_train), (x_validate, y_validate) = random_split(
            (x, y), (1.0 - validation_fraction, validation_fraction), rng=rng
        )

    # }}}

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)

    generator = torch.Generator(device=device)

    n = x_train.shape[0]
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    num_bad_epochs = 0

    for epoch in range(max_iter):
        # {{{ training

        model.train()
        perm = torch.randperm(n, generator=generator, device=device)

        loss_epoch = torch.zeros((), dtype=dtype, device=device)

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            n_batch = idx.shape[0]

            optimizer.zero_grad()

            # NOTE: match scikit-learn's L2 penalty, which only regularizes the
            # weight matrices and is scaled by the batch size, We can't use
            # torch's weight_decay here, because that regularizes differently
            loss = criterion(model(x_train[idx]), y_train[idx])
            loss = loss + 0.5 * alpha * _l2_penalty(model) / n_batch

            # update
            loss.backward()
            optimizer.step()

            # NOTE: weight each batch by its size so that the epoch loss matches
            # the (summed) loss reported by scikit-learn
            loss_epoch += loss.detach() * n_batch

        loss_train = loss_epoch / n

        # }}}

        # {{{ validation

        if x_validate is not None and y_validate is not None:
            model.eval()
            with torch.no_grad():
                loss_validate = criterion(model(x_validate), y_validate).item()

            log.info(
                "[%03d/%03d] train %.5e validate %.5e",
                epoch + 1,
                max_iter,
                loss_train,
                loss_validate,
            )

            if loss_validate < best_loss - tol:
                best_loss = loss_validate
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            if num_bad_epochs >= n_iter_no_change:
                log.info(
                    "Early stopping at epoch %03d (best %03d, loss %.5e)",
                    epoch + 1,
                    best_epoch,
                    best_loss,
                )
                break
        else:
            log.info("[%03d/%03d] train %.5e", epoch + 1, max_iter, loss_train)

        # }}}

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(
            "Restored best model from epoch %03d (loss %.5e)", best_epoch, best_loss
        )

    return MultilayerPerceptronClassifier(
        model, class_labels, dtype=dtype, device=device
    )


# }}}


# {{{ torch_train


def torch_train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    xy_train: TensorDataset,
    xy_validate: TensorDataset,
    *,
    epochs: int = 50,
    batch_size: int = 4096,
    target_scale: float = 1.0,
    callback: Callable[[int, nn.Module], None] | None = None,
    verbose: bool = False,
    patience_early_stop: int = 0,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.Tensor]:
    import torch

    # FIXME: not very nice to just split these?
    x_train, y_train = xy_train.tensors
    x_validate, y_validate = xy_validate.tensors
    device = x_train.device

    compiled_model = torch.compile(model)
    loader = SlicedDataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    num_bad_epochs = 0

    timer = TicTocTimer()
    losses = torch.empty((epochs, 2, 1), device=device)

    for epoch in range(epochs):
        # {{{ training

        model.train()
        loss_epoch = torch.tensor(0.0, device=device)
        num_batches = 0

        timer.tic()

        for i, (batch_x, batch_y) in enumerate(loader):
            optimizer.zero_grad()

            predictions = compiled_model(batch_x)
            loss = criterion(predictions, batch_y) / target_scale

            loss.backward()
            optimizer.step()

            loss_epoch += loss.detach()
            num_batches += 1

            if verbose:
                log.info(r"    \[batch %03d] loss %.5e", i, loss.item())

        loss_train = loss_epoch / num_batches

        # }}}

        # {{{ validation

        model.eval()
        with torch.no_grad():
            predictions = compiled_model(x_validate)
            loss_validate = criterion(predictions, y_validate) / target_scale

        # }}}

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss_validate)
        else:
            scheduler.step()

        timer.toc()

        log.info(
            "[%03d/%03d] train %.5e validate %.5e (%s)",
            epoch + 1,
            epochs,
            loss_train,
            loss_validate,
            timer.short(),
        )

        losses[epoch, 0] = loss_train.item()
        losses[epoch, 1] = loss_validate.item()

        # {{{ best model and early stopping

        loss_v = loss_validate.item()
        if loss_v < best_loss:
            best_loss = loss_v
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            num_bad_epochs = 0
        elif patience_early_stop > 0:
            num_bad_epochs += 1
            if num_bad_epochs >= patience_early_stop:
                log.info(
                    "Early stopping at epoch %03d (best %03d, loss %.5e)",
                    epoch + 1,
                    best_epoch,
                    best_loss,
                )
                losses = losses[: epoch + 1]
                break

        # }}}

        if callback is not None:
            with torch.no_grad():
                callback(epoch, compiled_model)  # ty: ignore[invalid-argument-type]

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(
            "Restored best model from epoch %03d (loss %.5e)", best_epoch, best_loss
        )

    return model, optimizer, losses


# }}}


# {{{ torch_predict


def torch_predict(
    model: nn.Module,
    *,
    bbox: tuple[float, float, float, float],
    resolution: int = 128,
    maxit: int = 128,
    device: Any = None,
) -> torch.Tensor:
    import array_api_compat
    import torch

    d = int(model.input_size)  # ty: ignore[invalid-argument-type]
    dtype = model.dtype
    assert isinstance(dtype, torch.dtype)
    assert dtype.is_floating_point

    from commonlib import generate_grid

    with torch.no_grad():
        z = torch.zeros((resolution, resolution, 2 * d), dtype=dtype, device=device)
        c = generate_grid(
            *bbox,
            nx=resolution,
            ny=resolution,
            dtype=dtype.to_complex(),
            xp=array_api_compat.array_namespace(z),
            device=device,
        )
        c = c[:, :, None]

        for _ in range(1, maxit):
            Az = model(z)
            z[:, :, ::2] = Az[:, :, ::2] + c.real
            z[:, :, 1::2] = Az[:, :, 1::2] + c.imag

    return torch.view_as_complex(z.reshape(*z.shape[:-1], -1, 2))


# }}}
