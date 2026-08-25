# ARCHITECTURE

```
src/tensorforge/
├── tensor.py    # Tensor + autodiff core + gradcheck harness (~320 LOC)
├── nn.py        # Linear, MLP, mse_loss, softmax_cross_entropy
├── optim.py     # SGD (momentum), Adam (bias-corrected)
└── cli.py       # gradcheck gate + train-xor demo (argparse only)
tests/test_tensorforge.py   # 20 tests
```

## Data flow
1. User builds a DAG of `Tensor` nodes via operator overloads; each op allocates an output
   Tensor holding a `_backward` closure capturing its parents and intermediate values.
2. `loss.backward()`: iterative DFS builds topological order from the loss node; seed
   dL/dL = 1; walk reverse-topo calling each node's `_backward`, which routes
   `out.grad` into parents with `_unbroadcast` shape reduction.
3. Optimizers read `.grad` directly on parameter Tensors (`zero_grad()` resets to None).

## Key correctness decisions
- **Unbroadcast reduction**: gradients are summed over broadcast-added leading dims and
  size-1 dims — the single most common source of wrong hand-rolled backward passes.
- **1-D matmul special cases**: `swapaxes(-1,-2)` is invalid for vectors; three explicit
  branches cover (M,N)@(N,), (N,)@(N,M) on both gradient sides.
- **gradcheck snapshots analytic grads** immediately after the first backward, because a
  user fn that calls backward internally would otherwise accumulate into the compared value.
- **Stable cross-entropy**: max-shifted logits + log-sum-exp; NLL via one-hot selection,
  normalized by batch size (a naive `.mean()` over the (B,K) product silently divides by K).

## Verification strategy
Every op is covered by central finite-difference checks across broadcasting shapes
(`(3,4)+(4,)`, `(3,1)*(1,4)`, `(2,3,4)+(4,)`). End-to-end learning proof: XOR MLP to 100%
accuracy inside pytest; CLI exit codes gate both gradcheck and training convergence.
