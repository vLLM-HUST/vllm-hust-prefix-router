"""CPU-only host doubles for CLI lifecycle tests; never installed in the wheel.

Only the two engine-dependent factories are replaced. The real JSON loader,
console script, HTTP service, connection pools and ZMQ supervisor still execute.
There is deliberately no publisher: readiness must stay degraded. This fixture
does not qualify native fingerprint compatibility, cache reuse or host activation.
"""

from vllm_hust_prefix_router.protocols.request_fingerprint import (
    UnsupportedFingerprintRequest,
)
from vllm_hust_prefix_router.service import config_loader


class CpuFingerprinter:
    def __init__(self, _config):
        pass

    async def fingerprint(self, _path, _payload):
        raise UnsupportedFingerprintRequest("CPU delivery fixture has no engine")


class CpuCodec:
    snapshot_replay = True

    def __init__(self, _version_range):
        pass

    def encode_replay_request(self, **_kwargs):
        return b"no-publisher"

    def decode_replay_response(self, _payload):
        raise AssertionError("no publisher is allowed in the CPU delivery fixture")


config_loader.VllmHustTextFingerprinter = CpuFingerprinter
config_loader.VllmHustKvEventCodec = CpuCodec
