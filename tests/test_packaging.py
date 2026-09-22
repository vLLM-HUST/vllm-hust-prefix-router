import json
import subprocess
import sys
from importlib import resources

from vllm_hust_prefix_router import __version__


def test_manifest_version_matches_distribution_version() -> None:
    manifest_path = resources.files("vllm_hust_prefix_router.manifests").joinpath(
        "vllm-hust-extension-v0.2.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["extension_id"] == "org.vllm-hust.prefix-router"
    assert manifest["extension_version"] == __version__
    assert manifest["kind"] == "control_plane_extension"
    assert manifest["runtime"]["type"] == "external_service"
    assert manifest["lifecycle_owner"] == "user"
    assert manifest["implementation"][0]["status"] == "import_only"


def test_package_import_does_not_import_runtime_dependencies() -> None:
    code = """
import sys
import vllm_hust_prefix_router
assert "aiohttp" not in sys.modules
assert "vllm" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
