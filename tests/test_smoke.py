"""Placeholder so CI proves the package installs and imports from a clean checkout."""

from __future__ import annotations


def test_package_imports() -> None:
    import vlm_swarm_coverage

    assert vlm_swarm_coverage.__version__
