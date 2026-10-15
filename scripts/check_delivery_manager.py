"""Check a disposable installed-wheel venv, including uninstall and native bypass.

Run with the target venv's Python, never a host/engine environment. Output is a
new directory; all commands are bounded and recorded. Uninstall is opt-in.
"""

import argparse
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

EXTENSION = "org.vllm-hust.prefix-router"
DISTRIBUTION = "vllm-hust-prefix-router"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uninstall-from-disposable-venv", action="store_true")
    args = parser.parse_args()
    if not args.uninstall_from_disposable_venv or sys.prefix == sys.base_prefix:
        parser.error(
            "an explicit disposable virtualenv uninstall acknowledgement is required"
        )
    # An engine installation is never an eligible disposable wheel environment.
    from importlib.metadata import PackageNotFoundError, version

    for package in ("vllm", "vllm-ascend", "torch", "torch-npu"):
        try:
            version(package)
        except PackageNotFoundError:
            continue
        parser.error(f"refusing to uninstall from an environment containing {package}")
    args.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    for key in ("CONFIG", "STATE", "CACHE"):
        env[f"XDG_{key}_HOME"] = str(args.output.resolve() / key.lower())
    suffix = ".exe" if os.name == "nt" else ""
    cli = str(Path(sysconfig.get_path("scripts")) / ("vllm-hust-ext" + suffix))
    records = []

    def run(label, command, *, expected=0, contains=None):
        result = subprocess.run(
            command,
            env=env,
            cwd=args.output,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        (args.output / f"{label}.log").write_text(output, encoding="utf-8")
        records.append({"label": label, "command": command, "exit": result.returncode})
        if result.returncode != expected or (contains and contains not in output):
            raise RuntimeError(f"{label} failed; inspect its retained log")
        return result.stdout

    def extension(action, **kwargs):
        return run(action, [cli, "extension", action, EXTENSION], **kwargs)

    probe = (
        "import os,sys; "
        "assert not any(k.startswith('PREFIX_ROUTER') for k in os.environ); "
        "assert 'vllm_hust_prefix_router' not in sys.modules; "
        "print('NATIVE_BYPASS_OK')"
    )

    def native(label):
        run(
            label,
            [cli, "run", "--", sys.executable, "-c", probe],
            contains="NATIVE_BYPASS_OK",
        )

    run("list", [cli, "extension", "list"], contains=f"{EXTENSION} 0.1.0.dev0 disabled")
    extension("inspect", contains="vllm.request-routing.front-door")
    extension("validate")
    extension("check", contains="descriptor-only")
    extension("plan", contains='"operation": "inspect_only"')
    extension("render", contains="inspect_only")
    extension("enable", expected=2, contains="implementation status: import_only")
    native("native-disabled")
    extension("disable", contains="disabled")
    run("disable-again", [cli, "extension", "disable", EXTENSION], contains="disabled")
    native("native-after-disable")
    extension("forget", contains="forgot")
    native("native-after-forget")
    # Forget removes state, not the installed distribution/discovery entry point.
    run("list-after-forget", [cli, "extension", "list"], contains=EXTENSION)
    run("uninstall", [sys.executable, "-m", "pip", "uninstall", "-y", DISTRIBUTION])
    discovered = run("list-after-uninstall", [cli, "extension", "list"])
    if EXTENSION in discovered:
        raise RuntimeError("extension remains discoverable after uninstall")
    run(
        "import-after-uninstall",
        [
            sys.executable,
            "-c",
            "import importlib.util; "
            "assert importlib.util.find_spec('vllm_hust_prefix_router') is None",
        ],
    )
    native("native-after-uninstall")
    run("pip-check", [sys.executable, "-m", "pip", "check"])
    (args.output / "result.json").write_text(
        json.dumps(
            {
                "result": "pass",
                "scope": "installed-wheel-manager-cpu",
                "automatic_activation": False,
                "commands": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
