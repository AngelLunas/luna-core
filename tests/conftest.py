"""Shared pytest fixtures for luna-core unit tests."""
from __future__ import annotations

import pytest

from luna_core.services.context_sources import clear_context_sources


@pytest.fixture(autouse=True)
def _isolate_context_source_registry():
    """The context source registry is process-global. Wipe it between tests so
    one test's `register_context_source` calls can't leak into the next.
    """
    clear_context_sources()
    yield
    clear_context_sources()
