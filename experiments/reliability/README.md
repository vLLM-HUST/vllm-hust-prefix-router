# Deterministic reliability regression

Only `control_fault_matrix.py` is retained from the research toolset. It exercises
the public Router with four fixed arrival traces, six faults and two admission
modes, using sequential HTTP requests, a virtual clock and synthetic KV usage.

See [delivery instructions](../../docs/reliability-delivery.md). Each new run has
its own output directory; `--validate` reconstructs its cell checks. Unit tests
reject forged hints, mismatched attempts, missing rows and false qualification
claims. This is not a real-NPU benchmark or engine-transfer fencing proof.

Historical real-model launchers, performance diagnostics and environment builders
remain in the separately preserved research archive, not this candidate tree.
Use their original frozen copies to inspect old evidence, never current code
inserted into an old run directory. [Evidence summary](../../docs/reliability-evidence.md)
states what has and has not been reproduced here.
