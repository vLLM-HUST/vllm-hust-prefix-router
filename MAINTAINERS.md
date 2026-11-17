# Maintainers

Original project owners:

- @Amber1qq — original implementation and final integration
- @WMASTER123 — runtime reliability and route fencing
- @Adr1anZheng — per-node load metrics

The vLLM-HUST organization provides repository infrastructure. Technical
direction, attribution, and release approval belong to the maintainers above.

## Contribution and verification records

| Maintainer | Contribution | Evidence |
| --- | --- | --- |
| @Amber1qq | Implemented the early router, integrated it into the legacy runtime, and archived the two-replica measurements. | [Legacy #80](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/80) (`7df266d1`), [legacy #173](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/173) (`6a71cd4b`), [benchmark #74](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/74). |
| @WMASTER123 | Hardened runtime failure handling, route fencing and lifecycle recovery in the standalone plugin. | [Legacy #225](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/225) (`5f7a84cc`), [plugin #5](https://github.com/vLLM-HUST/vllm-hust-prefix-router/pull/5) (`5ba9d710`), [benchmark #639 verification record](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/639). |
| @Adr1anZheng | Implemented per-node load metrics. The later load-signal study has its own experimental record. | [Legacy #258](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/258) (`19d650bc`), [plugin #8](https://github.com/vLLM-HUST/vllm-hust-prefix-router/pull/8), [benchmark #690](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/690). |

### Tested versions and release scope

- The historical dual-TP1 comparison used the integrated runtime at `68cf893a` and Ascend at
  `03a12f9b`, Qwen2.5-7B-Instruct BF16 and two 910B2 devices. The benchmark archive contains the per-run cache counters,
  first-token timing distributions and request rates.
  [Fixed comparison and counters](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/d4e688d6d841963e7d9f8ccade2b60abc33ad012/submissions/pr-154-merge-gates-prefix-routing-benefit-2chip-68cf893a4-03a12f9bdd/comparison.json).
- Standalone CPU, packaging and protocol verification, along with the native restart failure and
  remaining native recovery coverage, are recorded in the
  [reliability archive](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/70b28cc53b764e9555f341d7979c2a490208c219/archive/prefix-router-reliability-20261017/README.md).
- The November 15 eager diagnostic used Qwen3.5-35B-A3B BF16 on four 910B2 devices with two
  TP2+EP replicas. It tested an experimental wheel built from `5ba9d710` with the recorded
  random-seed and salt-hash changes. Its SHA-256 is
  `5a9d407b509d77d77a57ce9f197ac9c7712e9c5b67ad8aa06ad9e30d096feca4`. The measurements apply
  to this build and configuration. Qualification of an unmodified release wheel, workload quality
  and SLOs remain open.
  [Fixed nine-window report, software versions and original-file manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e46fd6dcc6e8d390c9193d988ca969b35384c9f1/reports/tech-outcomes-20261117/prefix-router/README.md).
