# TESTING

Run: `.venv/Scripts/python -m pytest tests -q` (bare — record real exit codes).

## Suite (20 tests)
| Area | Tests | Method |
|------|-------|--------|
| Op gradchecks vs finite differences | 9 | parametrized over broadcasting shapes; add/mul/matmul/exp/log/tanh/sigmoid/relu/div/pow/reductions/shape ops |
| Gradient accumulation (diamond graph) | 1 | analytic d(a³)/da |
| requires_grad gating | 1 | leaf without flag gets None grad |
| _unbroadcast unit | 1 | exact output shapes incl. scalar () |
| max tie semantics | 1 | PyTorch tie behavior + untied gradcheck |
| Linear layer | 1 | shapes + gradient flow to weight & bias |
| XOR end-to-end | 1 | Adam-trained MLP reaches accuracy 1.0 and >10× loss drop |
| SGD vs Adam regression | 1 | same step budget comparison |
| softmax cross-entropy | 1 | matches manual stable formula + finite-difference logit check |
| CLI | 3 | gradcheck exit 0 & no FAIL lines; train-xor exit 0 with accuracy 4/4 |

## Fixture-divergence discipline
Fixtures were chosen so the tested property actually varies (e.g., distinct values for max
gradchecks — max is non-differentiable at ties, which is asserted separately as tie *mass*
semantics instead). A first-pass bug where the CLI re-randomized a constant tensor inside
the finite-difference lambda was caught by exactly this discipline.

## Results (2026-08-26)
- pytest: 20 passed, exit 0
- ruff check .: clean, exit 0
- bandit -r src -ll: clean, exit 0
- Live CLI run: train-xor --verbose → loss 0.3278 → 0.000000, accuracy 4/4, exit 0
