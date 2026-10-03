"""Shared pytest fixtures: tiny synthetic RGB-T sequences, dummy configs."""

import pytest


@pytest.fixture
def dummy_config() -> dict:
    """Minimal config dict for unit tests that do not need real data."""
    return {
        "model": {
            "backbone": "vitb_shared",
            "img_size": 256,
            "template_size": 128,
            "num_identity_tokens": 16,
            "num_parity_tokens": 16,
            "num_variable_nodes": 256,
            "bp_iterations": 2,
        }
    }
