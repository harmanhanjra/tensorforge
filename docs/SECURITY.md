# SECURITY

See THREAT_MODEL.md for the full analysis. Summary:

- **No network I/O. No subprocesses. No eval/exec. No file reads/writes. No env secrets.**
- CLI uses argparse with fixed subcommands; arguments never reach dynamic execution.
- Single runtime dependency (numpy); no install-time network code beyond standard packaging.
- Bandit scan: clean at medium+ severity (exit 0 with `-ll`). B101 eliminated by replacing
  an input-validation assert with `TypeError`.
- Input validation: non-scalar powers raise TypeError; allocation size is caller-controlled
  by design (library), CLI only creates tiny demo tensors.
- Residual accepted risks are documented in THREAT_MODEL.md (finite-difference tolerances;
  numpy trusted as arithmetic substrate).
