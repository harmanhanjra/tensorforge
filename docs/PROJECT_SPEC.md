# PROJECT SPEC — tensorforge (Cycle 8)

Niche: Neural Networks × Automatic Differentiation × Developer Verification Tooling.
Difficulty: 7 (broadcasting-correct reverse-mode AD is a known-hard correctness problem;
Cycle 3 was difficulty 4, Cycle 7 was 6).

## Problem
Learners and practitioners hand-writing autograd engines have no automated proof their
backward passes are correct; broadcasting makes hand-derived reductions error-prone.
Existing engines (micrograd et al.) are scalar-only or ship unverified gradients.

## Functional requirements
FR1 `Tensor` wraps float64 ndarray; tracks DAG via `_prev`; `requires_grad` gating.
FR2 Ops (each defines forward + backward closure):
    add, sub, mul, div, pow(scalar), matmul, neg, exp, log, tanh, sigmoid, relu,
    sum, mean, max, reshape, transpose, broadcasting everywhere applicable.
FR3 `Tensor.backward()` seeds d(out)/d(out)=1 and propagates in reverse topological order.
FR4 Gradients reduce correctly under NumPy broadcasting (leading dims + size-1 dims).
FR5 nn: `Linear(in,out)` (x@W+b), `mse_loss`, `softmax_cross_entropy` (max-shifted stable).
FR6 optim: `SGD` (momentum optional), `Adam` (bias-corrected), shared `zero_grad`.
FR7 `gradcheck(fn, *inputs, eps, tol)` central-difference verification returning bool.
FR8 CLI: `tensorforge gradcheck [--verbose]` (exit 1 on failure),
    `tensorforge train-xor` (prints per-epoch loss, final accuracy).
FR9 Determinism: seeded `randn` helpers.

## Non-goals
GPU/CUDA, autograd for control flow, vjp/jvp duals, serialization, Python <3.9.

## Acceptance criteria
- pytest suite green: op gradchecks (incl. broadcast shapes like (3,1)+(4,), matmul chains),
  XOR training converges to 100%, Adam beats SGD-on-step-count sanity, topo-order diamond graph.
- ruff clean-ish (documented exceptions), bandit no high/medium findings.
- Live run: `tensorforge train-xor` shows monotone-ish loss decrease over real execution.
