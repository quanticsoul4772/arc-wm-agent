import pytest


@pytest.fixture(autouse=True)
def _restore_only_reset_levels(monkeypatch):
    """Engine._make sets ONLY_RESET_LEVELS for its whole process (the real gateway is another
    process); restore it after each test so it cannot reach the in-process gateway simulator."""
    monkeypatch.delenv("ONLY_RESET_LEVELS", raising=False)
