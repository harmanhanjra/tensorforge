# tensorforge

A from-scratch **reverse-mode automatic differentiation engine** over NumPy arrays —
multi-dimensional tensors, broadcasting-correct gradient reduction, a small neural-net
library, SGD/Adam optimizers, and a first-class **numerical gradient-verification harness**
(`gradcheck`) usable as a library API or CI exit-code gate.

Most educational autograd engines stop at scalars (micrograd) or ship backward passes nobody
can verify. tensorforge's core idea: *the verification harness is part of the engine*, not an
afterthought — every op's analytic gradient is continuously proven correct against central
finite differences.

## Features
- `Tensor` class with reverse-mode autodiff over `float64` ndarrays
- Ops: `+ - * / ** @ exp log tanh sigmoid relu sum mean max reshape transpose` — all
  broadcasting-aware (gradients are correctly *unbroadcast* back to parameter shapes)
- `nn`: `Linear`, MLP composition, `mse_loss`, stable softmax cross-entropy
- `optim`: SGD and Adam (bias-corrected)
- `gradcheck(fn, *tensors)` + CLI: `tensorforge gradcheck` exits non-zero on any mismatch
- End-to-end proof: a 2→8→1 tanh MLP trained on XOR reaches 4/4 inside the test suite

## Install & test
```bash
uv venv .venv && uv pip install -e ".[test]" --python .venv
.venv/Scripts/python -m pytest tests -q        # Windows/git-bash
# or: .venv/bin/python -m pytest tests -q
```

## CLI
```bash
tensorforge gradcheck --verbose     # verifies every op against finite differences; exit 0 = pass
tensorforge train-xor              # trains an MLP on XOR, prints loss per epoch
```

## Quick start
```python
from tensorforge import Tensor
from tensorforge.nn import Linear, mse_loss
from tensorforge.optim import Adam

W = Tensor.randn(3, 1, requires_grad=True)
b = Tensor.zeros(1, requires_grad=True)
x = Tensor.randn(16, 3)
y = x.data @ W.data + b.data          # synthetic target

opt = Adam([W, b], lr=0.05)
for step in range(200):
    pred = x @ W + b
    loss = mse_loss(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
print(loss.item())                    # → ~0
```

## Docs
See [docs/](docs/): PROJECT_SPEC · ARCHITECTURE · THREAT_MODEL · SECURITY · TESTING · WHY_THIS_PROJECT.

## License
MIT — see [LICENSE](LICENSE).
