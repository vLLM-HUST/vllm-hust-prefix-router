"""Reject unqualified builds and cache identities without requiring a host."""

import hashlib
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from vllm_hust_prefix_router.adapters.vllm_hust import hust_native_profile as profile
from vllm_hust_prefix_router.service.config_loader import load_runtime_config


@dataclass
class Stored:
    block_hashes: tuple = (b"a" * 32,)
    medium: str = "GPU"
    group_idx: int = 0
    locality: str | None = None
    ownership: str | None = None
    block_size: int = 128
    kv_cache_spec_kind: str = "full_attention"
    kv_cache_spec_sliding_window: int | None = None
    lora_id: int | None = None
    lora_name: str | None = None
    extra_keys: list | None = None


@dataclass
class Removed:
    block_hashes: tuple = (b"a" * 32,)
    medium: str = "GPU"
    group_idx: int = 0
    locality: str | None = None
    ownership: str | None = None


class Cleared:
    pass


def decode(events, rank=0):
    return profile.normalize_hust_batch(
        SimpleNamespace(events=events, data_parallel_rank=rank),
        stored=Stored,
        removed=Removed,
        cleared=Cleared,
    )


def test_dense_local_cache_preserves_store_remove_clear():
    batch = decode([Stored(), Removed(), Cleared()])
    assert batch.data_parallel_rank == 0
    assert batch.events[0].block_hashes == (b"a" * 32,)
    assert batch.events[1].block_hashes == (b"a" * 32,)
    assert type(batch.events[2]).__name__ == "AllBlocksCleared"


@pytest.mark.parametrize(
    "field,value",
    [
        ("medium", "CPU"),
        ("medium", None),
        ("group_idx", 1),
        ("group_idx", None),
        ("locality", "REMOTE"),
        ("ownership", "tier"),
        ("block_size", 64),
        ("kv_cache_spec_kind", "sliding_window"),
        ("kv_cache_spec_kind", None),
        ("kv_cache_spec_sliding_window", 128),
        ("lora_id", 1),
        ("lora_name", "adapter"),
        ("extra_keys", [("salt",)]),
        ("block_hashes", (42,)),
        ("block_hashes", (b"short",)),
    ],
)
def test_unqualified_cache_identity_rejected(field, value):
    with pytest.raises(ValueError):
        decode([replace(Stored(), **{field: value})])


@pytest.mark.parametrize(
    "field,value",
    [
        ("medium", "CPU"),
        ("group_idx", 1),
        ("locality", "REMOTE"),
        ("ownership", "tier"),
        ("block_hashes", (1,)),
    ],
)
def test_removal_from_another_tier_cannot_erase_local_blocks(field, value):
    with pytest.raises(ValueError):
        decode([replace(Removed(), **{field: value})])


@pytest.mark.parametrize("rank", [None, 1])
def test_unqualified_rank_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        decode([Stored()], rank)


def install_fixture(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "0")
    expected = {}
    packages = {}
    for name, version in profile.PACKAGE_VERSIONS.items():
        root = tmp_path / name.replace("-", "_")
        root.mkdir()
        (root / "interface.py").write_bytes(b"canonical interface")
        expected[name] = {
            "interface.py": hashlib.sha256(b"canonical interface").hexdigest()
        }
        packages[name] = SimpleNamespace(
            version=version, locate_file=lambda _name, root=root: root
        )
    monkeypatch.setattr(profile, "distribution", lambda name: packages[name])
    monkeypatch.setattr(profile, "SOURCE_HASHES", expected)
    return packages


def test_matching_pins_and_installed_interface_pass(monkeypatch, tmp_path):
    install_fixture(monkeypatch, tmp_path)
    assert profile.verify_hust_native_profile() == profile.PACKAGE_VERSIONS


@pytest.mark.parametrize("fault", ["core", "ascend", "bytes", "seed", "hashes"])
def test_build_or_hash_drift_is_rejected_before_host_import(
    monkeypatch, tmp_path, fault
):
    packages = install_fixture(monkeypatch, tmp_path)
    if fault == "core":
        packages["vllm"].version = "0.30.1"
    elif fault == "ascend":
        packages["vllm-ascend"].version = "0.25.1rc2.dev805"
    elif fault == "bytes":
        (tmp_path / "vllm/interface.py").write_bytes(b"changed")
    elif fault == "seed":
        monkeypatch.setenv("PYTHONHASHSEED", "1")
    else:
        monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    with pytest.raises(RuntimeError):
        profile.verify_hust_native_profile()


def test_cross_profile_model_identity_is_distinct(monkeypatch):
    monkeypatch.setattr(profile, "legacy_model_identity", lambda _model: "a" * 64)
    assert profile.model_identity("/model") != "a" * 64


def test_new_profile_configuration_uses_exact_pins_and_existing_replay(
    monkeypatch, tmp_path
):
    import json

    from vllm_hust_prefix_router.service import config_loader

    class Codec:
        pass

    monkeypatch.setattr(profile, "HustNativeKvEventCodec", Codec)
    monkeypatch.setattr(profile, "model_identity", lambda _model: "a" * 64)
    captured = []
    monkeypatch.setattr(
        config_loader,
        "VllmHustTextFingerprinter",
        lambda cfg: captured.append(cfg) or object(),
    )
    config = {
        "runtime_profile": profile.PROFILE,
        "hash_block_size": 128,
        "fingerprint": {"tokenizer": "/model"},
        "backends": [
            {
                "id": "worker",
                "url": "http://127.0.0.1:18000",
                "data_parallel_rank": 0,
                "managed_lease_path": "/lease",
            }
        ],
    }
    path = tmp_path / "router.json"
    path.write_text(json.dumps(config))
    service = load_runtime_config(path).service
    assert captured[0].host_version_range == "==0.30.0+empty"
    assert service.event_sources[0].codec.__class__ is Codec
    config["fingerprint"]["host_version_range"] = ">=0.23"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="omit host_version_range"):
        load_runtime_config(path)
