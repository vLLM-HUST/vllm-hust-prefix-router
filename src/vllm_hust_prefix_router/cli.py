"""Command-line entry point kept import-safe for Extension Manager discovery."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vllm_hust_prefix_router._version import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing runtime dependencies."""
    parser = argparse.ArgumentParser(prog="vllm-hust-prefix-router")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="run the external router service")
    serve.add_argument("--config", required=True, help="JSON configuration path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command."""
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        from aiohttp import web

        from vllm_hust_prefix_router.service.app import create_app
        from vllm_hust_prefix_router.service.config_loader import load_runtime_config

        runtime = load_runtime_config(args.config)
        web.run_app(create_app(runtime.service), host=runtime.host, port=runtime.port)
    return 0
