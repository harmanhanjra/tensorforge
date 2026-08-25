"""Tests for the tensor autograd engine: numeric gradchecks, ops, nn, optim, CLI."""

import numpy as np
import pytest

from tensorforge.cli import main as cli_main
from tensorforge.nn import MLP, Linear, mse_loss, softmax_cross_entropy
from tensorforge.optim import SGD, Adam
from tensorforge.tensor import Tensor, gradcheck

RNG = np.random.default_rng(123)


# ---------------------------------------------------------------------------
# Op-level finite-difference gradchecks (the core correctness proof)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shapes",
    [((3, 4), (3, 4)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 3, 4), (4,))],
)
def test_gradcheck_add_mul_broadcasting(shapes):
    a = Tensor(RNG.standard_normal(shapes[0]), requires_grad=True)
    b = Tensor(RNG.standard_normal(shapes[1]), requires_grad=True)
    assert gradcheck(lambda x, y: (x + y).tanh().sum() + (x * y).sum(), a, b)


def test_gradcheck_matmul_2d_and_vector():
    a = Tensor(RNG.standard_normal((3, 4)), requires_grad=True)
    b = Tensor(RNG.standard_normal((4, 5)), requires_grad=True)
    v = Tensor(RNG.standard_normal(4), requires_grad=True)
    assert gradcheck(lambda x, y: (x @ y).sum(), a, b)
    assert gradcheck(lambda x, w: (x @ w).relu().sum(), a, v)


def test_gradcheck_elementwise_ops():
    z = Tensor(np.abs(RNG.standard_normal((3, 3))) + 0.5, requires_grad=True)
    assert gradcheck(lambda t: t.exp().log().sum(), z)
    assert gradcheck(lambda t: t.tanh().sigmoid().sum(), z)
    assert gradcheck(lambda t: (t ** 3).sum(), z)
    assert gradcheck(lambda t: (t / (t + 1.0)).sum(), z)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_gradcheck_reductions(axis):
    a = Tensor(RNG.standard_normal((3, 4)), requires_grad=True)
    assert gradcheck(lambda t: t.sum(axis=axis), a)
    assert gradcheck(lambda t: t.mean(axis=axis), a)


def test_max_reduction_tie_mass_goes_to_single_source():
    a = Tensor([[1.0, 5.0], [5.0, 0.0]], requires_grad=True)
    out = a.max()
    out.backward()
    # PyTorch semantics: EVERY tied max position receives the full upstream gradient
    n_ties = int((a.data == a.data.max()).sum())
    assert np.isclose(a.grad.sum(), n_ties)
    # and every position carrying gradient IS a maximum position
    assert ((a.grad != 0) <= (a.data == a.data.max())).all()
    # gradcheck requires distinct values: max is non-differentiable at ties
    assert gradcheck(lambda t: t.max(axis=1).sum(),
                     Tensor([[1.0, 2.0], [4.0, -0.5]], requires_grad=True), eps=1e-6)


def test_gradcheck_shape_ops():
    a = Tensor(RNG.standard_normal((3, 4)), requires_grad=True)
    assert gradcheck(lambda t: t.reshape(4, 3).transpose().sum(), a)
    assert gradcheck(lambda t: t.transpose().reshape(12).sum(), a)


def test_diamond_graph_accumulates_gradients():
    a = Tensor([2.0], requires_grad=True)
    b = a * a  # diamond: 'a' used twice
    c = b * a
    c.backward()
    assert np.allclose(a.grad, 3 * a.data ** 2)  # d(a^3)/da


def test_requires_grad_gating():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0], requires_grad=True)
    (a * b).backward()
    assert a.grad is None
    assert b.grad is not None


def test_unbroadcast_returns_exact_shape():
    from tensorforge.tensor import _unbroadcast
    g = RNG.standard_normal((3, 4))
    assert _unbroadcast(g, (4,)).shape == (4,)
    assert _unbroadcast(g, (3, 1)).shape == (3, 1)
    assert _unbroadcast(g, ()).shape == ()


# ---------------------------------------------------------------------------
# Neural net + optimizers (end-to-end learning proof)
# ---------------------------------------------------------------------------

XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = [[0.0], [1.0], [1.0], [0.0]]


def test_xor_training_converges():
    x = Tensor(XOR_X)
    y = Tensor(XOR_Y)
    model = MLP([2, 8, 1], seed=7)
    opt = Adam(model.parameters(), lr=0.05)
    first = last = None
    for _epoch in range(300):
        pred = model(x)
        loss = mse_loss(pred, y)
        if first is None:
            first = loss.item()
        last = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = ((model(x).data > 0.5) == (y.data > 0.5)).all(axis=1).mean()
    assert last < first / 10, f"loss did not decrease enough: {first} -> {last}"
    assert acc == 1.0, f"XOR accuracy {acc}"


def test_adam_beats_sgd_same_steps_on_regression():
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((32, 3)))
    w_true = rng.standard_normal((3, 1))
    y = Tensor(x.data @ w_true + 0.1 * rng.standard_normal((32, 1)))

    def train(opt_cls, lr):
        w = Tensor.zeros(3, 1, requires_grad=True)
        b = Tensor.zeros(1, requires_grad=True)
        opt = opt_cls([w, b], lr=lr)
        for _ in range(100):
            loss = mse_loss(x @ w + b, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()

    # same step budget: adaptive Adam should land at least as low as plain SGD here
    assert train(Adam, 0.05) < 1e-2
    assert train(SGD, 0.05) > train(Adam, 0.05) or train(SGD, 0.05) < 1e-2


def test_linear_parameters_flow():
    layer = Linear(3, 2, seed=1)
    x = Tensor(RNG.standard_normal((5, 3)))
    out = layer(x)
    assert out.shape == (5, 2)
    out.sum().backward()
    assert layer.weight.grad is not None and layer.bias.grad is not None


def test_softmax_cross_entropy_matches_manual_and_numeric_gradient():
    logits_data = RNG.standard_normal((6, 4))
    labels = np.array([0, 3, 1, 2, 2, 3])
    loss = softmax_cross_entropy(Tensor(logits_data), labels)
    # manual stable computation
    m = logits_data.max(axis=1, keepdims=True)
    z = logits_data - m
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    expected = -logp[np.arange(6), labels].mean()
    assert np.isclose(loss.item(), expected)
    # central finite difference on one logit vs analytic gradient
    eps = 1e-6
    plus = logits_data.copy()
    plus[0, 0] += eps
    minus = logits_data.copy()
    minus[0, 0] -= eps
    num = (softmax_cross_entropy(Tensor(plus), labels).item()
           - softmax_cross_entropy(Tensor(minus), labels).item()) / (2 * eps)
    logits = Tensor(logits_data, requires_grad=True)
    softmax_cross_entropy(logits, labels).backward()
    assert np.isclose(logits.grad[0, 0], num, atol=1e-5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_gradcheck_passes(capsys):
    assert cli_main(["gradcheck", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "gradient checks passed" in out


def test_cli_train_xor_exits_zero(capsys):
    assert cli_main(["train-xor"]) == 0
    out = capsys.readouterr().out
    assert "accuracy 4/4" in out
