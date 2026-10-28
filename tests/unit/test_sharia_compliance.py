"""Deterministic numerical, evidence, and provider tests; no live finance calls."""

import json
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from execution.sharia_compliance import (
    POLICY,
    FinancialSnapshot,
    ReviewedFileProvider,
    ShariaScreener,
    Status,
    YahooSnapshotProvider,
    is_sharia_compliant,
    screen_snapshot,
    screener_from_config,
)

TODAY = date(2026, 10, 27)


@pytest.fixture
def snapshot():
    return FinancialSnapshot(
        symbol='TEST',
        as_of=date(2026, 6, 30),
        source='Reviewed annual filing, note 12',
        currency='USD',
        total_assets=1000,
        total_debt=100,
        cash=100,
        interest_bearing_investments=50,
        receivables=100,
        revenue=1000,
        interest_income=10,
        non_compliant_revenue=20,
        business_reviewed=True,
    )


def screen(snapshot, **changes):
    return screen_snapshot(replace(snapshot, **changes), today=TODAY)


def test_complete_evidence_passes_with_audit_details(snapshot):
    result = screen(snapshot)
    assert result.status == Status.COMPLIANT
    assert result.compliant
    assert result.policy == POLICY
    assert result.source == snapshot.source
    assert result.as_of == snapshot.as_of
    assert result.ratios == {
        'debt_to_assets': 0.1,
        'cash_and_interest_to_assets': 0.15,
        'cash_and_receivables_to_assets': 0.2,
        'non_permissible_income_to_revenue': 0.03,
    }


@pytest.mark.parametrize(
    ('changes', 'ratio'),
    [
        ({'total_debt': 334}, 'debt_to_assets'),
        ({'interest_bearing_investments': 234}, 'cash_and_interest_to_assets'),
        ({'receivables': 401}, 'cash_and_receivables_to_assets'),
        ({'non_compliant_revenue': 41}, 'non_permissible_income_to_revenue'),
    ],
)
def test_each_screen_can_independently_fail(snapshot, changes, ratio):
    result = screen(snapshot, **changes)
    assert result.status == Status.NON_COMPLIANT
    assert len(result.reasons) == 1
    assert ratio in result.reasons[0]


@pytest.mark.parametrize(
    ('changes', 'passes'),
    [
        ({'total_debt': '333.32999'}, True),
        ({'total_debt': '333.33000'}, False),
        ({'total_debt': '333.33001'}, False),
        ({'interest_bearing_investments': '233.32999'}, True),
        ({'interest_bearing_investments': '233.33000'}, False),
        ({'interest_bearing_investments': '233.33001'}, False),
        ({'receivables': '399.99999'}, True),
        ({'receivables': '400'}, False),
        ({'receivables': '400.00001'}, False),
        ({'non_compliant_revenue': '39.99999'}, True),
        ({'non_compliant_revenue': '40'}, True),
        ({'non_compliant_revenue': '40.00001'}, False),
    ],
)
def test_exact_threshold_boundaries(snapshot, changes, passes):
    assert screen(snapshot, **changes).compliant is passes


@pytest.mark.parametrize(
    'name',
    [
        'total_assets',
        'total_debt',
        'cash',
        'interest_bearing_investments',
        'receivables',
        'revenue',
        'interest_income',
        'non_compliant_revenue',
    ],
)
@pytest.mark.parametrize('bad', [None, -1, float('nan'), float('inf'), '-Infinity', True, 'bad'])
def test_missing_or_invalid_values_never_pass(snapshot, name, bad):
    assert screen(snapshot, **{name: bad}).status == Status.INSUFFICIENT_DATA


@pytest.mark.parametrize('name', ['total_assets', 'revenue'])
def test_zero_denominators(snapshot, name):
    assert screen(snapshot, **{name: 0}).status == Status.INSUFFICIENT_DATA


def test_explicit_zero_amounts_are_valid(snapshot):
    assert screen(snapshot, total_debt=0, cash=0, interest_bearing_investments=0, receivables=0, interest_income=0, non_compliant_revenue=0).compliant


@pytest.mark.parametrize(
    'changes',
    [
        {'as_of': TODAY - timedelta(days=451)},
        {'as_of': TODAY + timedelta(days=1)},
        {'source': ''},
        {'currency': ''},
        {'currency': 'usd'},
        {'business_reviewed': False},
        {'business_reviewed': 'true'},
        {'cash': 900},
        {'non_compliant_revenue': 1001},
    ],
)
def test_evidence_quality_checks(snapshot, changes):
    assert screen(snapshot, **changes).status == Status.INSUFFICIENT_DATA


def test_maximum_age_boundary(snapshot):
    assert screen(snapshot, as_of=TODAY - timedelta(days=450)).compliant


def test_missing_data_takes_precedence_over_financial_failure(snapshot):
    assert screen(snapshot, total_debt=900, non_compliant_revenue=None).status == Status.INSUFFICIENT_DATA


@pytest.mark.parametrize('industry', ['Banks—Diversified', 'banks - diversified', 'Software', '', 'Gambling'])
def test_industry_labels_cannot_replace_revenue_evidence(snapshot, industry):
    assert screen(snapshot, industry=industry, business_reviewed=False).status == Status.INSUFFICIENT_DATA


def test_all_failures_are_reported(snapshot):
    result = screen(snapshot, total_debt=700, cash=350, receivables=200, non_compliant_revenue=100)
    assert len(result.reasons) == 4


def test_provider_normalization_and_symbol_binding(snapshot):
    provider = Mock()
    provider.fetch.return_value = snapshot
    screener = ShariaScreener(provider)
    assert screener.check(' test ', today=TODAY).compliant
    provider.fetch.assert_called_with('TEST')
    assert screener.check('OTHER', today=TODAY).status == Status.INSUFFICIENT_DATA


@pytest.mark.parametrize('symbol', ['', '   ', '../TEST', None, 'A B'])
def test_invalid_symbols_do_not_call_provider(symbol):
    provider = Mock()
    assert not ShariaScreener(provider).check(symbol).compliant
    provider.fetch.assert_not_called()


def test_provider_errors_fail_closed():
    provider = Mock()
    provider.fetch.side_effect = RuntimeError('rate limited')
    assert ShariaScreener(provider).check('TEST').status == Status.INSUFFICIENT_DATA


@pytest.mark.parametrize('age', [0, -1, 451, True, '450'])
def test_invalid_maximum_age_rejected(age):
    with pytest.raises(ValueError):
        ShariaScreener(max_age_days=age)


def test_unknown_methodology_rejected():
    with pytest.raises(ValueError):
        screener_from_config({'methodology': 'unknown'})


def test_reviewed_file_reload_and_precision(tmp_path, snapshot):
    path = tmp_path / 'reviewed.json'
    data = {k: v for k, v in snapshot.__dict__.items() if k != 'symbol'}
    data['as_of'] = snapshot.as_of.isoformat()
    data['total_debt'] = '333.32999999'
    path.write_text(json.dumps({'TEST': data}), encoding='utf-8')
    screener = screener_from_config({'evidence_file': str(path)})
    assert screener.check('TEST', today=TODAY).compliant
    data['total_debt'] = '333.33000000'
    path.write_text(json.dumps({'TEST': data}), encoding='utf-8')
    assert screener.check('TEST', today=TODAY).status == Status.NON_COMPLIANT
    assert screener.check('MISSING', today=TODAY).status == Status.INSUFFICIENT_DATA
    path.write_text('{invalid')
    assert screener.check('TEST', today=TODAY).status == Status.INSUFFICIENT_DATA


def test_file_provider_preserves_decimal(tmp_path):
    path = tmp_path / 'evidence.json'
    path.write_text('{"TEST": {"as_of":"2026-06-30", "source":"filing", "currency":"USD", "cash":0.1234567890123456789}}')
    assert ReviewedFileProvider(path).fetch('TEST').cash == Decimal('0.1234567890123456789')


@pytest.fixture
def yahoo():
    with patch('execution.sharia_compliance.yf.Ticker') as factory:
        ticker = factory.return_value
        ticker.info = {'financialCurrency': 'USD', 'industry': 'Software', 'debtToEquity': 0}
        latest, older = pd.Timestamp('2026-06-30'), pd.Timestamp('2025-06-30')
        # Deliberately unsorted to catch positional-column selection.
        ticker.balance_sheet = pd.DataFrame(
            {older: [800, 0, 50, 20], latest: [1000, 100, 100, 100]},
            index=['Total Assets', 'Total Debt', 'Cash And Cash Equivalents', 'Accounts Receivable'],
        )
        ticker.income_stmt = pd.DataFrame({latest: [1000, 10]}, index=['Total Revenue', 'Interest Income'])
        yield ticker


def test_yahoo_uses_latest_matching_fiscal_period_and_no_quote_ratios(yahoo):
    snapshot = YahooSnapshotProvider().fetch('TEST')
    assert snapshot.total_assets == 1000
    assert snapshot.total_debt == 100
    assert snapshot.as_of == date(2026, 6, 30)
    assert snapshot.interest_bearing_investments is None
    assert snapshot.non_compliant_revenue is None
    assert not snapshot.business_reviewed
    assert screen(snapshot).status == Status.INSUFFICIENT_DATA
    assert not is_sharia_compliant('TEST')


def test_yahoo_does_not_fall_back_to_an_old_complete_period(yahoo):
    yahoo.income_stmt.columns = [pd.Timestamp('2025-06-30')]
    assert ShariaScreener().check('TEST', today=TODAY).status == Status.INSUFFICIENT_DATA


def test_yahoo_missing_statement(yahoo):
    yahoo.balance_sheet = pd.DataFrame()
    assert not ShariaScreener().check('TEST', today=TODAY).compliant


def test_yahoo_missing_interest_income_is_not_zero(yahoo):
    yahoo.income_stmt = yahoo.income_stmt.drop('Interest Income')
    assert YahooSnapshotProvider().fetch('TEST').interest_income is None
