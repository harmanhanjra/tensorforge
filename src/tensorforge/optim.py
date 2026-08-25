"""Optimizers: SGD and Adam over lists of parameter Tensors."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor], lr: float):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    """Gradient descent with optional momentum."""

    def __init__(self, params, lr=0.05, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self._vel = {id(p): np.zeros_like(p.data) for p in self.params}

    def step(self) -> None:
        for p in self.params:
            v = self._vel[id(p)]
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with bias correction."""

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self._m = {id(p): np.zeros_like(p.data) for p in self.params}
        self._v = {id(p): np.zeros_like(p.data) for p in self.params}

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        for p in self.params:
            g = p.grad
            m, v = self._m[id(p)], self._v[id(p)]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
