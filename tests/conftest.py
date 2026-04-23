"""Shared fixtures for thermographic-report-builder tests."""
import os
import pytest

# Set env vars at module level so Settings() can instantiate during collection.
# The autouse fixture below re-applies them per-test via monkeypatch for clean isolation.
os.environ.setdefault("SOLAR_PROJECT_ID", "test-project-001")
os.environ.setdefault("SOLAR_USER_ID", "test-user-001")
os.environ.setdefault("ENVIRONMENT", "test")


@pytest.fixture(autouse=True)
def set_required_env(monkeypatch):
    """Set minimum required environment variables for Settings to instantiate."""
    monkeypatch.setenv("SOLAR_PROJECT_ID", "test-project-001")
    monkeypatch.setenv("SOLAR_USER_ID", "test-user-001")
    monkeypatch.setenv("ENVIRONMENT", "test")


@pytest.fixture
def sample_panel_boxes():
    """Sample panel bounding boxes for testing."""
    from thermographic_report_builder.models.defect import BoundingBox

    return [
        BoundingBox(left=100, top=200, width=50, height=80, label="solarpanels"),
        BoundingBox(left=160, top=200, width=50, height=80, label="solarpanels"),
        BoundingBox(left=100, top=290, width=50, height=80, label="solarpanels"),
    ]


@pytest.fixture
def sample_defect_boxes():
    """Sample defect bounding boxes for testing."""
    from thermographic_report_builder.models.defect import BoundingBox

    return [
        BoundingBox(left=110, top=220, width=10, height=15, label="hotspots"),
        BoundingBox(left=170, top=230, width=8, height=12, label="faultydiodes"),
    ]
