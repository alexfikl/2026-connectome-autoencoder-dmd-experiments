# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

# {{{ loss functions


class ComplexMSELoss(nn.Module):
    def forward(self, y_hat, y):
        dy = y_hat - y
        return (dy.conj() * dy).real.mean()


# }}}


# {{{ activation functions


class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x * x


class LeakyQuadraticActivation(nn.Module):
    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()

        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (1.0 - self.alpha) * x * x


class ComplexTanhActivation(nn.Module):
    def forward(self, x):
        return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


# }}}


# {{{ layers


class BiasLayer(nn.Module):
    def __init__(self, size: int, *, dtype: Any = None) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class ComplexLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: Any | None = None,
    ):
        super().__init__()

        ftype = None if dtype is None else dtype.to_real()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=ftype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # kaiming uniform (same as nn.Linear default) as a safe fallback
        stdv = 1.0 / np.sqrt(self.weight.shape[1])
        nn.init.uniform_(self.weight, -stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            nn.functional.linear(x.real, self.weight),
            nn.functional.linear(x.imag, self.weight),
        )


class ComplexQuadratic(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        dtype: Any = None,
    ) -> None:
        if output_size is None:
            output_size = input_size

        if dtype is None:
            dtype = torch.complex64

        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype

        self.model = nn.Sequential(
            ComplexLinear(input_size, output_size, dtype=dtype),
            QuadraticActivation(),
            BiasLayer(output_size, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class QuadraticBias(nn.Module):
    def __init__(self, size: int, dtype: Any = None) -> None:
        super().__init__()

        self.input_size = size
        self.model = nn.Sequential(QuadraticActivation(), BiasLayer(size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# }}}

# {{{ init


def init_xavier_uniform_(
    x: torch.Tensor,
    *,
    nonlinearity: str = "quadratic",
    param: float | None = None,
) -> None:
    if nonlinearity == "quadratic":
        stdv = 1.0 / x.shape[1] ** 0.25
        nn.init.uniform_(x, -stdv, stdv)
    else:
        gain = nn.init.calculate_gain(nonlinearity, param=param)  # ty: ignore[invalid-argument-type]
        nn.init.xavier_uniform_(x, gain=gain)


# }}}
