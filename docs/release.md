# PyPI release

The package name is `vllm-hust-prefix-router`. PyPI ownership is assigned in
PyPI's project settings; it cannot be set by `pyproject.toml`. The target owner
is the `intellistream` organization at <https://pypi.org/org/intellistream/>.

Before the first upload, a member of that organization must create the
`vllm-hust-prefix-router` project under **intellistream / Projects**. In the
new project's **Publishing** settings, add a GitHub Actions Trusted Publisher:

| Field | Value |
|---|---|
| Repository owner | `vLLM-HUST` |
| Repository | `vllm-hust-prefix-router` |
| Workflow | `publish-pypi.yml` |
| Environment | `pypi` |

The release workflow uses GitHub OIDC; no long-lived PyPI token belongs in
GitHub secrets, shell history, or this repository. The first release tag is
`v0.1.0a1`. Only the canonical GitHub repository can publish it.

## Source exposure guard

The release workflow builds one CPython-specific wheel each for Python
3.10 through 3.14. It compiles package `.py` files to adjacent `.pyc` files,
removes the `.py` files, updates wheel tags and RECORD, then tests installation
from the resulting wheel in a clean environment. Before upload, a separate
job rejects any source distribution, unexpected file, missing wheel version,
or `.py` / `.pyi` member. **Do not upload the ordinary wheel or an sdist.**

Bytecode is not encryption and can be reverse engineered. The GitHub source
repository remains public by project decision; this guard limits direct source
disclosure through PyPI release artifacts.

Before creating the tag, verify on the target environment:

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m build --wheel --outdir build-dist
python scripts/build_protected_wheel.py build-dist/*.whl dist
```

The protected wheel's Manager smoke must pass `list`, `inspect`, `validate`,
`check`, `plan`, `render`, `enable`, `disable`, and `forget`. A managed launch
must fail closed without bound service health. Real TP2/NPU correctness remains
a separate deployment qualification; this alpha release makes no performance
claim.

After the tag workflow completes, verify the release at
<https://pypi.org/project/vllm-hust-prefix-router/> and that its owner is the
`intellistream` organization. Confirm that PyPI lists only five CPython wheels
and no source distribution. Install the exact version in a clean environment,
check the extension registration, and retain wheel SHA-256 checksums as
release evidence.
