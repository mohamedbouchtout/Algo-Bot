"""Integration tests for the Scheduler."""

import pytest


@pytest.mark.integration
class TestSchedulerIntegration:
    """Basic smoke tests for the Scheduler (no IB required)."""

    def test_is_weekend_returns_bool(self):
        from core.scheduler import Scheduler

        sched = Scheduler()
        result = sched.is_weekend()
        assert isinstance(result, bool)

    def test_is_market_hours_returns_bool(self):
        from core.scheduler import Scheduler

        sched = Scheduler()
        result = sched.is_market_hours()
        assert isinstance(result, bool)
