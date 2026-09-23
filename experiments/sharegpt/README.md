# ShareGPT Prefix/Lifecycle evidence matrix

This directory runs a controlled, paired comparison using the frozen
ShareGPT-derived workload described by the project experiment handoff.

The formal matrix is three repetitions at 2, 8, 16, 24, 32, 40, and 48
requests per second. Every Prefix/Lifecycle point uses the same prompt order
and Poisson arrival seed. Each point starts four fresh workers and a fresh
external Router. The first policy alternates within each `(repetition, rate)`
pair to reduce run-order bias.

The four workers must remain on one explicit, fixed device list for the whole
matrix. Every backend, including node0, uses HTTP through the Router. Each
backend receives an independent pool of 512 connections and an observable
bounded queue. A result is valid only if the workload and manifest hashes,
prompt order, input token lengths, output lengths, event-source readiness,
worker placement, plugin commit, and wheel hash are captured by `evidence.json`.

Run `run_point.py` directly for startup acceptance. Run `run_matrix.py` for
the resumable formal matrix. The manager skips only successful evidence for
the same policy, repetition, and offered rate. `summarize.py` produces raw
point CSV, paired point CSV, and per-rate medians; paired ratios are always
`Lifecycle / Prefix`.

This single-host four-worker matrix is not multi-machine evidence. A genuine
multi-machine validation must place backends on at least two physical hosts,
record each host identity and accelerator mapping, and keep the Router on an
explicit host. It must not label multiple processes or containers on one host
as multi-machine.
