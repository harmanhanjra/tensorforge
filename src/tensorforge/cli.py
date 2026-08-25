"""tensorforge CLI: gradcheck verification gate + XOR training demo."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .nn import MLP, mse_loss
from .optim import Adam
from .tensor import Tensor, gradcheck


def _run_gradcheck(verbose: bool) -> int:
    """Verify analytic gradients of representative ops against finite differences."""
    rng = np.random.default_rng(42)

    def check(name, fn, *inputs, **kw):
        ok = gradcheck(fn, *inputs, **kw)
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
        return ok

    a = Tensor(rng.standard_normal((3, 4)), requires_grad=True)
    b = Tensor(rng.standard_normal((4,)), requires_grad=True)
    c = Tensor(rng.standard_normal((3, 1)), requires_grad=True)
    w2 = Tensor(rng.standard_normal((4, 2)), requires_grad=True)
    b2 = Tensor(rng.standard_normal((3, 2)), requires_grad=True)
    s = Tensor(np.abs(rng.standard_normal((3, 3))) + 0.5, requires_grad=True)
    results = [
        check("add+broadcast (3,4)+(4,)", lambda x, y: x + y, a, b),
        check("mul+broadcast (3,4)*(3,1)", lambda x, y: x * y, a, c),
        check("matmul (3,4)@(4,) ", lambda x, y: x @ y, a, b),
        check("chain matmul+tanh", lambda x, y: ((x @ w2) + y).tanh().sum(), a, b2),
        check("exp/log", lambda z: z.log().exp().sum(), s),
        check("sigmoid/relu", lambda z: z.sigmoid().relu().sum(), a),
        check("div/pow", lambda z: (z / (z * z)).sum(), s),
        check("max reduction", lambda z: z.max(axis=1).sum(), a),
        check("mean reduction", lambda z: z.mean(), a),
        check("reshape+transpose", lambda z: z.reshape(4, 3).transpose().sum(), a),
    ]
    failed = sum(1 for r in results if not r)
    if verbose:
        print(f"{len(results) - failed}/{len(results)} gradient checks passed")
    return 1 if failed else 0


def _train_xor(verbose: bool) -> int:
    x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = Tensor([[0.0], [1.0], [1.0], [0.0]])
    model = MLP([2, 8, 1], seed=7)
    opt = Adam(model.parameters(), lr=0.05)
    losses = []
    for epoch in range(300):
        pred = model(x)
        loss = mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if verbose and (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1:4d}  loss {loss.item():.6f}")
    pred = model(x)
    correct = int(((pred.data > 0.5) == (y.data > 0.5)).all(axis=1).sum())
    final = losses[-1]
    print(f"final loss {final:.6f}  accuracy {correct}/4")
    if verbose:
        trend = "decreased" if final < losses[0] else "DID NOT decrease"
        print(f"loss {trend}: {losses[0]:.4f} -> {final:.4f}")
    return 0 if (correct == 4 and final < losses[0]) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tensorforge",
                                     description="NumPy autograd engine with gradcheck")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gradcheck", help="verify gradients via finite differences")
    g.add_argument("--verbose", action="store_true")
    t = sub.add_parser("train-xor", help="train an MLP on XOR")
    t.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "gradcheck":
        return _run_gradcheck(args.verbose)
    return _train_xor(args.verbose)


if __name__ == "__main__":
    sys.exit(main())
