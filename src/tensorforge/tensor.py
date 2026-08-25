"""Core Tensor class: reverse-mode automatic differentiation over NumPy ndarrays.

Design notes
------------
- Each Tensor holds `data` (float64 ndarray), optional `.grad` (same shape), a tuple of
  parent Tensors `_prev`, and a closure `_backward` that routes output grad to parents.
- Broadcasting: forward ops use plain NumPy broadcasting; backward uses `_unbroadcast`
  to sum gradients back down to each parent's shape (leading dims + size-1 dims).
- backward() walks the DAG in reverse topological order (iterative DFS post-order).
"""

from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` so it fits `shape` (inverse of NumPy broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev: tuple[Tensor, ...] = _children
        self._op = _op

    # -- construction helpers -------------------------------------------------
    @staticmethod
    def randn(*shape, requires_grad=False, seed=None) -> "Tensor":
        rng = np.random.default_rng(seed)
        return Tensor(rng.standard_normal(shape), requires_grad=requires_grad)

    @staticmethod
    def zeros(*shape, requires_grad=False) -> "Tensor":
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def item(self) -> float:
        return float(self.data)

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad}, op={self._op!r})"

    # -- autodiff machinery ---------------------------------------------------
    def backward(self) -> None:
        """Backpropagate from this scalar-or-array tensor through the DAG."""
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(t: Tensor) -> None:
            if id(t) in visited:
                return
            visited.add(id(t))
            for child in t._prev:
                build(child)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    # -- arithmetic ops --------------------------------------------------------
    def __add__(self, other) -> "Tensor":
        other = _ensure(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                     (self, other), "+")

        def _bw():
            g = out.grad
            if self.requires_grad:
                self.grad = _acc(self.grad, _unbroadcast(g, self.data.shape))
            if other.requires_grad:
                other.grad = _acc(other.grad, _unbroadcast(g, other.data.shape))
        out._backward = _bw
        return out

    def __mul__(self, other) -> "Tensor":
        other = _ensure(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                     (self, other), "*")

        def _bw():
            g = out.grad
            if self.requires_grad:
                self.grad = _acc(self.grad, _unbroadcast(g * other.data, self.data.shape))
            if other.requires_grad:
                other.grad = _acc(other.grad, _unbroadcast(g * self.data, other.data.shape))
        out._backward = _bw
        return out

    def __truediv__(self, other) -> "Tensor":
        other = _ensure(other)
        return self * other**-1

    def __pow__(self, p: float) -> "Tensor":
        if not isinstance(p, (int, float)):
            raise TypeError("only scalar powers supported")
        out = Tensor(self.data ** p, self.requires_grad, (self,), f"**{p}")

        def _bw():
            if self.requires_grad:
                self.grad = _acc(self.grad, out.grad * p * self.data ** (p - 1))
        out._backward = _bw
        return out

    def __matmul__(self, other) -> "Tensor":
        other = _ensure(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad,
                     (self, other), "@")

        def _bw():
            g = out.grad
            a, b = self.data, other.data
            if self.requires_grad:
                if b.ndim == 1:      # (...,M,N) @ (N,) -> ga = g[...,None] * b
                    ga = np.expand_dims(g, -1) * b
                elif a.ndim == 1:    # (N,) @ (N,M) -> ga = outer(g, b)
                    ga = np.outer(g, b)
                else:
                    ga = g @ b.swapaxes(-1, -2)
                self.grad = _acc(self.grad, _unbroadcast(ga, a.shape))
            if other.requires_grad:
                if a.ndim == 1:      # (N,) @ (N,M) -> gb = outer(a, g)
                    gb = np.outer(a, g)
                elif b.ndim == 1:    # (M,N) @ (N,) -> gb = a.T @ g
                    gb = a.swapaxes(-1, -2) @ g
                else:
                    gb = a.swapaxes(-1, -2) @ g
                other.grad = _acc(other.grad, _unbroadcast(gb, b.shape))
        out._backward = _bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return _ensure(other) + (-self)

    # -- activations / elementwise ---------------------------------------------
    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), self.requires_grad, (self,), "exp")

        def _bw():
            if self.requires_grad:
                self.grad = _acc(self.grad, out.grad * out.data)
        out._backward = _bw
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")

        def _bw():
            if self.requires_grad:
                self.grad = _acc(self.grad, out.grad / self.data)
        out._backward = _bw
        return out

    def tanh(self) -> "Tensor":
        out = Tensor(np.tanh(self.data), self.requires_grad, (self,), "tanh")

        def _bw():
            if self.requires_grad:
                self.grad = _acc(self.grad, out.grad * (1.0 - out.data ** 2))
        out._backward = _bw
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")

        def _bw():
            if self.requires_grad:
                self.grad = _acc(self.grad, out.grad * s * (1.0 - s))
        out._backward = _bw
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")

        def _bw():
            if self.requires_grad:
                self.grad = _acc(self.grad, out.grad * (self.data > 0))
        out._backward = _bw
        return out

    # -- reductions -------------------------------------------------------------
    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad,
                     (self,), "sum")

        def _bw():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self.grad = _acc(self.grad, np.broadcast_to(g, self.data.shape).copy())
        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def max(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), self.requires_grad,
                     (self,), "max")
        mask = (self.data == self.data.max(axis=axis, keepdims=True))

        def _bw():
            if self.requires_grad:
                g = out.grad if keepdims else \
                    (np.expand_dims(out.grad, axis) if axis is not None else out.grad)
                self.grad = _acc(self.grad, (g * mask).astype(np.float64))
        out._backward = _bw
        return out

    # -- shape ops ---------------------------------------------------------------
    def reshape(self, *shape) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _bw():
            if self.requires_grad:
                self.grad = _acc(self.grad, out.grad.reshape(self.data.shape))
        out._backward = _bw
        return out

    def transpose(self) -> "Tensor":
        out = Tensor(self.data.T, self.requires_grad, (self,), "T")

        def _bw():
            if self.requires_grad:
                self.grad = _acc(self.grad, out.grad.T)
        out._backward = _bw
        return out


def _ensure(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _acc(current, incoming):
    return incoming if current is None else current + incoming


def gradcheck(fn, *inputs, eps: float = 1e-6, tol: float = 1e-5) -> bool:
    """Verify analytic gradients of fn(*inputs) against central finite differences.

    Only inputs with requires_grad=True are checked. Returns True when every element
    of every checked input matches within `tol` (relative to max(1, |numeric|)).
    """
    inputs = [_ensure(t) for t in inputs]
    # Fresh-slate gradients: repeated gradchecks on shared tensors must not inherit
    # accumulated .grad from previous calls.
    for t in inputs:
        t.grad = None
    out = fn(*inputs)
    if not isinstance(out, Tensor):  # allow python scalars via reductions
        out = Tensor(out)
    if out.grad is None:
        out.backward()
    # Snapshot analytic gradients NOW: user-supplied fn may call backward() again
    # during the numeric loop below, which would accumulate into .grad.
    analytic = {id(t): None if t.grad is None else t.grad.copy() for t in inputs}
    ok = True
    for t in inputs:
        if not t.requires_grad:
            continue
        num = np.zeros_like(t.data)
        it = np.nditer(t.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = t.data[idx]
            t.data[idx] = orig + eps
            lp = float(np.asarray(fn(*inputs).data).sum())
            t.data[idx] = orig - eps
            lm = float(np.asarray(fn(*inputs).data).sum())
            t.data[idx] = orig
            num[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        ana = analytic[id(t)]
        if ana is None or not np.allclose(ana, num, atol=tol, rtol=tol):
            ok = False
    return ok
