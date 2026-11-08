# Reliability evidence scope

This unreleased reliability increment is based on the public lifecycle Router
and adds atomic cache recovery, managed engine identity checks, request cleanup
and bounded diagnostics. Public acceptance is limited to source, package and
deterministic CPU checks that can be rerun from this repository.

The complete historical measurement summary, artifact digests, retained failure
and custody limitations are maintained in the centralized
[vllm-hust-benchmark evidence capsule](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/main/archive/prefix-router-reliability-20261017).
That capsule records owner-held native observations; it does not contain the raw
archives and therefore does not establish formal NPU or performance qualification.

## Reproducible gates

Use the commands in [the delivery guide](reliability-delivery.md) against the
reviewed source, an unpacked source distribution and an isolated installed
wheel. Keep skips separate from passes. The repository CI covers supported
Python versions, formatting, dependency checks, package contents and Extension
Manager discovery, validation, lifecycle rollback and uninstall behavior.

The deterministic fault matrix uses synthetic backends, a virtual clock and
sequential requests. It verifies state transitions and failure handling, not
concurrent load, tail latency, throughput or accelerator behavior.

## Qualification boundary

The following remain outside this increment:

- engine-atomic pre-reuse and KV-byte transfer fences;
- automatic external-service startup;
- real multi-DP routing and broad host-version compatibility;
- isolated reliability overhead, throughput, tail-latency or speedup claims;
- complete final-wheel native fault replay and reviewer-accessible raw evidence.

Installation and Manager enablement record configuration intent only. A managed
launch remains fail-closed until configured Router, KV subscription and backend
health checks pass. Runtime effectiveness requires separate owning-process
request evidence; health alone is insufficient.

Protect the diagnostic endpoint on a private interface or at the deployment
edge. Its bounded receipts are operational diagnostics, not a durable audit log.
