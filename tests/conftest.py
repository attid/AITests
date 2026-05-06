"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def non_mocked_hosts() -> list[str]:
    return []


@pytest.fixture(autouse=True)
def _allow_unused_responses(httpx_mock):
    """Allow tests to register more responses than they consume."""
    # pytest-httpx >=0.30 stores options on _options; older versions on .options.
    # Try the more recent API first.
    options = getattr(httpx_mock, "_options", None) or getattr(httpx_mock, "options", None)
    if options is not None and hasattr(options, "assert_all_responses_were_requested"):
        options.assert_all_responses_were_requested = False
    yield
