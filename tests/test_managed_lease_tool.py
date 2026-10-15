"""Operator lease publication must refuse overwrite and unsafe directories."""

import importlib.util
import os
from pathlib import Path

import pytest

tool_path = Path(__file__).resolve().parents[1] / "scripts/write_managed_lease.py"
if not tool_path.is_file():
    pytest.skip("source/sdist operator helper", allow_module_level=True)
spec = importlib.util.spec_from_file_location("lease_tool", tool_path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class Lease:
    def __init__(self, current=True):
        self.current = current

    def is_current(self):
        return self.current

    def to_dict(self):
        return {"generation": "fresh-fixture"}


@pytest.mark.skipif(os.name != "posix", reason="Linux operator filesystem contract")
def test_publication_is_private_and_does_not_overwrite_existing_lease(tmp_path):
    tmp_path.chmod(0o700)
    path = tmp_path / "lease.json"
    tool.publish(Lease(), path)
    original = path.read_bytes()
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        tool.publish(Lease(), path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "posix", reason="Linux operator filesystem contract")
@pytest.mark.parametrize("mode", [0o777, 0o770])
def test_publication_refuses_shared_write_directory(tmp_path, mode):
    tmp_path.chmod(mode)
    with pytest.raises(ValueError, match="writable"):
        tool.publish(Lease(), tmp_path / "lease.json")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="Linux operator filesystem contract")
def test_dead_identity_is_not_published(tmp_path):
    tmp_path.chmod(0o700)
    with pytest.raises(ProcessLookupError):
        tool.publish(Lease(False), tmp_path / "lease.json")
    assert list(tmp_path.iterdir()) == []
