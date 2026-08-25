# THREAT MODEL — tensorforge

## Scope
Offline numerical library + CLI. No network I/O, no subprocesses, no secrets, no persistence
beyond user-invoked output. The attack surface is intentionally minimal; this document exists
to prove that was a deliberate, reviewed decision.

## Assets
1. User data passed into `Tensor` (numerical arrays — potentially proprietary model data).
2. Integrity of gradient computations (silent wrong gradients = corrupted downstream models).
3. Host filesystem (CLI writes nothing by default).

## Trust boundaries
- Caller code → tensorforge API (same process, trusted caller assumed; library never exfiltrates,
  logs, or persists input values).
- CLI arguments → engine (argv only; no file reads, no env-var secrets, no URLs).

## Threats & mitigations
| ID | Threat | Mitigation |
|----|--------|-----------|
| T1 | Silent wrong-gradient bugs corrupting research/training | First-class `gradcheck` finite-difference harness; full test suite asserts analytic == numeric for every op incl. broadcasting edge cases |
| T2 | Numeric instability (overflow in exp/softmax) producing inf/NaN silently | Stable formulations: max-shifted softmax, log-sum-exp in cross-entropy; documented behavior |
| T3 | DoS via pathological shapes (memory blow-up on huge tensors) | Library-level: documented that allocation is caller-controlled; no auto-loading of files; CLI creates only tiny demo tensors (≤ 4×8) |
| T4 | Supply-chain: dependency compromise | Single runtime dep (numpy); test deps pinned loosely but auditable; no install-time code beyond hatchling build |
| T5 | Code injection through CLI args | argparse with fixed choices; no eval/exec/subprocess anywhere (bandit-verified) |
| T6 | Accidental overwrite of user files | CLI prints to stdout only; no `--output` flag exists |

## Non-goals / out of scope
- Multi-user serving, authentication, encryption (no I/O to protect)
- Distributed training security

## Residual risks (accepted)
- R1: float64 finite-difference checks use fixed tolerances (1e-5 rel); extremely large-magnitude
  inputs could produce false failures — documented, configurable via `tol` parameter.
- R2: numpy itself is trusted as the arithmetic substrate.
