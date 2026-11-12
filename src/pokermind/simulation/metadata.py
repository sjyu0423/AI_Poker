"""Reproducibility metadata recorded alongside every simulation run."""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import pokermind


def _git_info(cwd: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def collect_metadata(command: list[str] | None = None) -> dict[str, Any]:
    """Environment information. Not part of the deterministic result."""
    return {
        "pokermind_version": pokermind.__version__,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "command": command if command is not None else sys.argv,
        "git": _git_info(Path(__file__).resolve().parent),
    }
