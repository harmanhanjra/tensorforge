"""Tiny neural-net library on top of the Tensor autograd engine."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # Xavier/Glorot-style init keeps tanh MLPs in a healthy regime.
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(rng.standard_normal((in_features, out_features)) * scale,
                             requires_grad=True)
        self.bias = Tensor.zeros(out_features, requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    d = pred - target
    return (d * d).mean()


def softmax_cross_entropy(logits: Tensor, labels) -> Tensor:
    """Max-shifted stable softmax NLL over the last axis; `labels` is an int array."""
    m = logits.max(axis=1, keepdims=True)
    z = logits - m
    logsumexp = z.exp().sum(axis=1, keepdims=True).log()
    log_probs = z - logsumexp
    onehot = np.zeros(logits.shape)
    onehot[np.arange(logits.shape[0]), np.asarray(labels).astype(int)] = 1.0
    n = logits.shape[0]
    return -(log_probs * Tensor(onehot)).sum() * (1.0 / n)


class MLP:
    def __init__(self, sizes: list[int], seed: int | None = None):
        layers = []
        for i in range(len(sizes) - 1):
            s = None if seed is None else seed + i
            layers.append(Linear(sizes[i], sizes[i + 1], seed=s))
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.tanh()
        return x

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]
