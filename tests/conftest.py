"""Shared fixtures for the M8A application tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokermind.app.context import AppContext, load_app_context

HERO = Path("runs/m4-leduc-teacher/strategies/strategy_100000.json")


@pytest.fixture(scope="session")
def app_ctx() -> AppContext:
    """The frozen research context (needs the git-ignored M4 teacher strategy)."""
    if not HERO.is_file():
        pytest.skip("M4 teacher strategy not available (git-ignored run output)")
    return load_app_context()
