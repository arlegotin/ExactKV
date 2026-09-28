from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-metal", action="store_true", default=False)
    parser.addoption("--run-model", action="store_true", default=False)
    parser.addoption("--run-slow", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    permissions = {
        "metal": config.getoption("--run-metal"),
        "model": config.getoption("--run-model"),
        "slow": config.getoption("--run-slow"),
    }
    for item in items:
        for marker, allowed in permissions.items():
            if item.get_closest_marker(marker) and not allowed:
                item.add_marker(pytest.mark.skip(reason=f"requires --run-{marker}"))
