"""Exercise real screening through strategy scanning and the broker entry boundary."""

import asyncio
import json
from copy import deepcopy
from datetime import date
from unittest.mock import Mock, patch

import pytest

from execution.sharia_compliance import FinancialSnapshot, ShariaScreener
from tests.conftest import CONFIG, PARAMS, make_synthetic_bars


@pytest.fixture(scope='module')
def order_manager_class():
    """Give ib_insync/eventkit an explicit loop before importing them.

    Python 3.14 no longer creates a default loop implicitly. Importing at
    collection time fails before fixtures can establish the broker's context.
    Only close a loop owned by this fixture; preserve an existing caller's loop.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    owns_loop = loop is None or loop.is_closed()
    if owns_loop:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        from execution.order_manager import OrderManager

        yield OrderManager
    finally:
        if owns_loop:
            loop.close()
            asyncio.set_event_loop(None)


@pytest.fixture
def manager(order_manager_class):
    params = deepcopy(PARAMS)
    params['sharia_compliance'] = {'enabled': True}
    ib = Mock()
    positions = Mock(active_positions={})
    analyzer = Mock()
    stock_data = Mock()
    stock_data.get_historical_data.return_value = make_synthetic_bars(300)
    manager = order_manager_class(ib, stock_data, positions, Mock(), {'Technology': analyzer}, CONFIG, params)
    provider = Mock()
    provider.fetch.return_value = FinancialSnapshot(
        symbol='TEST',
        as_of=date.today(),
        source='Reviewed filing',
        currency='USD',
        total_assets=1000,
        total_debt=100,
        cash=100,
        interest_bearing_investments=0,
        receivables=100,
        revenue=1000,
        interest_income=0,
        non_compliant_revenue=0,
        business_reviewed=True,
    )
    manager.sharia_screener = ShariaScreener(provider)
    return manager


def signal(direction='LONG'):
    return {'symbol': 'TEST', 'type': direction, 'entry': 100, 'stop': 95, 'target': 110, 'breakout_volume_ratio': 2, 'retest_volume_ratio': 0.3}


@pytest.mark.integration
class TestShariaOrderFlow:
    @pytest.mark.parametrize('entry', ['execute_signal', 'place_order'])
    @pytest.mark.parametrize('direction', ['SHORT', 'SELL', '', 'long'])
    def test_non_long_entries_blocked_before_broker_access(self, manager, entry, direction):
        args = (signal(direction), 10) if entry == 'place_order' else (signal(direction),)
        getattr(manager, entry)(*args)
        assert manager.ib.mock_calls == []
        manager.sharia_screener.provider.fetch.assert_not_called()

    @pytest.mark.parametrize('entry', ['execute_signal', 'place_order'])
    @pytest.mark.parametrize('evidence', ['non_compliant', 'missing', 'error'])
    def test_failed_or_unknown_evidence_blocks_all_entry_paths(self, manager, entry, evidence):
        from dataclasses import replace

        provider = manager.sharia_screener.provider
        if evidence == 'error':
            provider.fetch.side_effect = OSError('offline')
        else:
            provider.fetch.return_value = replace(provider.fetch.return_value, total_debt=800 if evidence == 'non_compliant' else None)
        args = (signal(), 10) if entry == 'place_order' else (signal(),)
        getattr(manager, entry)(*args)
        assert manager.ib.mock_calls == []

    def test_approved_long_reaches_order_submission(self, manager):
        manager.ib.accountSummary.return_value = [Mock(tag='NetLiquidation', value='100000')]
        manager.ib.portfolio.return_value = []
        with patch.object(manager, 'place_order') as submit:
            manager.execute_signal(signal())
        submit.assert_called_once()

    def test_direct_approved_long_reaches_broker(self, manager):
        manager.ib.reqTickers.return_value = [Mock(marketPrice=Mock(return_value=None))]
        manager.place_order(signal(), 10)
        manager.ib.qualifyContracts.assert_called_once()
        manager.ib.placeOrder.assert_not_called()

    def test_disabled_mode_preserves_short_trading(self, manager):
        manager.sharia_screener = None
        manager.ib.accountSummary.return_value = [Mock(tag='NetLiquidation', value='100000')]
        manager.ib.portfolio.return_value = []
        with patch.object(manager, 'place_order') as submit:
            manager.execute_signal(signal('SHORT'))
        submit.assert_called_once()

    @pytest.mark.parametrize('strategy', ['ai', 'technical'])
    @pytest.mark.parametrize('direction', ['LONG', 'SHORT'])
    def test_both_scan_strategies_obey_entry_guard(self, manager, strategy, direction):
        analyzer = manager.ai_analyzers['Technology']
        analyzer.predict.return_value = {'class': direction, 'probs': {direction: 0.99}} if strategy == 'ai' else None
        analyzer.construct_signal.return_value = signal(direction)
        manager.ib.accountSummary.return_value = [Mock(tag='NetLiquidation', value='100000')]
        manager.ib.portfolio.return_value = []
        with patch('execution.order_manager.TrendIndicator') as technical, patch.object(manager, 'place_order') as submit:
            technical.return_value.detect_breakout_and_retest.return_value = signal(direction)
            manager.scan_stocks({'Technology': {'Software': ['TEST']}})
        assert submit.call_count == (1 if direction == 'LONG' else 0)

    def test_revocation_between_sizing_and_submission_blocks_order(self, manager):
        from dataclasses import replace

        provider = manager.sharia_screener.provider
        approved = provider.fetch.return_value
        provider.fetch.side_effect = [approved, replace(approved, non_compliant_revenue=100)]
        manager.ib.accountSummary.return_value = [Mock(tag='NetLiquidation', value='100000')]
        manager.ib.portfolio.return_value = []
        manager.execute_signal(signal())
        manager.ib.qualifyContracts.assert_not_called()
        manager.ib.placeOrder.assert_not_called()
        assert provider.fetch.call_count == 2


@pytest.mark.integration
def test_configured_file_evidence_reaches_order_sizing(tmp_path, order_manager_class):
    """Actual configuration -> JSON provider -> ratios -> order sizing."""
    path = tmp_path / 'evidence.json'
    path.write_text(
        json.dumps(
            {
                'TEST': {
                    'as_of': date.today().isoformat(),
                    'source': 'Reviewed annual filing',
                    'currency': 'USD',
                    'total_assets': 1000,
                    'total_debt': 100,
                    'cash': 100,
                    'interest_bearing_investments': 0,
                    'receivables': 100,
                    'revenue': 1000,
                    'interest_income': 0,
                    'non_compliant_revenue': 0,
                    'business_reviewed': True,
                }
            }
        ),
        encoding='utf-8',
    )
    params = deepcopy(PARAMS)
    params['sharia_compliance'] = {'enabled': True, 'evidence_file': str(path)}
    ib = Mock()
    ib.accountSummary.return_value = [Mock(tag='NetLiquidation', value='100000')]
    ib.portfolio.return_value = []
    manager = order_manager_class(ib, Mock(), Mock(active_positions={}), Mock(), {}, CONFIG, params)
    with patch.object(manager, 'place_order') as submit:
        manager.execute_signal(signal())
        submit.assert_called_once()
        path.write_text('{}', encoding='utf-8')
        submit.reset_mock()
        manager.execute_signal(signal())
        submit.assert_not_called()


@pytest.mark.integration
def test_disabled_configuration_never_constructs_screener(order_manager_class):
    params = deepcopy(PARAMS)
    params['sharia_compliance'] = {'enabled': False, 'methodology': 'unused'}
    with patch('execution.order_manager.screener_from_config') as factory:
        manager = order_manager_class(Mock(), Mock(), Mock(), Mock(), {}, CONFIG, params)
    assert manager.sharia_screener is None
    factory.assert_not_called()
