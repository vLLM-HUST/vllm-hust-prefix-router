"""Replace Python sources in a wheel with CPython bytecode for PyPI release.

The resulting wheel is interpreter-specific. It keeps package metadata,
entry points, manifests, and license files, but contains no ``.py`` files.
Bytecode reduces direct source disclosure; it is not encryption.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import py_compile
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def build_protected_wheel(wheel: Path, output_dir: Path) -> Path:
    if sys.implementation.name != "cpython":
        raise RuntimeError("protected wheels require CPython")
    parts = wheel.name.rsplit("-", 3)
    stem, original_tag = parts[0], "-".join(parts[1:])
    if original_tag != "py3-none-any.whl":
        raise ValueError(f"expected a py3-none-any wheel, got {wheel.name}")
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}-none-any"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{stem}-{tag}.whl"
    if output.exists():
        raise FileExistsError(output)

    with tempfile.TemporaryDirectory(prefix="protected-wheel-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            for item in archive.namelist():
                path = PurePosixPath(item)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe wheel member: {item}")
            archive.extractall(root)

        sources = list(root.rglob("*.py"))
        if not sources:
            raise ValueError("wheel has no Python sources to protect")
        for source in sources:
            relative = source.relative_to(root).as_posix()
            if not relative.startswith("vllm_hust_prefix_router/"):
                raise ValueError(f"unexpected Python source: {relative}")
            py_compile.compile(
                str(source),
                cfile=str(source.with_suffix(".pyc")),
                dfile=relative,
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
            )
            source.unlink()

        dist_info = list(root.glob("*.dist-info"))
        if len(dist_info) != 1:
            raise ValueError("wheel must contain one dist-info directory")
        wheel_metadata = dist_info[0] / "WHEEL"
        metadata = wheel_metadata.read_text(encoding="utf-8")
        if metadata.count("Tag: py3-none-any") != 1:
            raise ValueError("wheel metadata has an unexpected tag")
        wheel_metadata.write_text(
            metadata.replace("Tag: py3-none-any", f"Tag: {tag}"),
            encoding="utf-8",
        )

        record = dist_info[0] / "RECORD"
        with record.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path == record:
                    continue
                data = path.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                writer.writerow(
                    (
                        path.relative_to(root).as_posix(),
                        f"sha256={digest.decode().rstrip('=')}",
                        len(data),
                    )
                )
            writer.writerow((record.relative_to(root).as_posix(), "", ""))

        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())

    with zipfile.ZipFile(output) as archive:
        if any(name.endswith(".py") for name in archive.namelist()):
            output.unlink()
            raise RuntimeError("protected wheel still contains Python sources")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(build_protected_wheel(args.wheel, args.output_dir))


if __name__ == "__main__":
    main()
