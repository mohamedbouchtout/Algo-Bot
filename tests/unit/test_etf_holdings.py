"""Holdings eligibility, dated snapshots, outage handling and cache refresh."""

import csv
import io
from datetime import date, timedelta
from unittest.mock import Mock, patch

import pytest
import requests

from execution.etf_holdings import ETF_POLICY, ETFHoldingsScreener, parse_holdings
from execution.sharia_compliance import Status, screener_from_config

TODAY = date(2026, 11, 16)


def feed(fund, symbols=(), day=TODAY, extra=()):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Account', 'StockTicker', 'SecurityName', 'Shares', 'Price', 'MarketValue', 'MoneyMarketFlag'])
    for symbol in [*(f'ZZ{chr(65 + i)}' for i in range(26)), *symbols]:
        writer.writerow([day.strftime('%m/%d/%Y'), fund, symbol, 'Example Corp', 10, 5, 50, ''])
    writer.writerows(extra)
    return output.getvalue()


def response(text):
    return Mock(text=text)


@pytest.mark.parametrize('matches', [('HLAL',), ('SPUS',), ('HLAL', 'SPUS')])
def test_membership_in_either_fund_is_enough(matches):
    def get(url, **kwargs):
        fund = 'SPUS' if 'SPUS' in url else 'HLAL'
        return response(feed(fund, ['TEST'] if fund in matches else []))

    with patch('execution.etf_holdings.requests.get', side_effect=get):
        result = ETFHoldingsScreener().check('test', today=TODAY)
    assert result.compliant
    assert result.policy == ETF_POLICY
    assert result.as_of == TODAY
    for fund in matches:
        assert fund in result.source


def test_no_membership_blocks():
    with patch('execution.etf_holdings.requests.get', side_effect=[response(feed('HLAL')), response(feed('SPUS'))]):
        result = ETFHoldingsScreener().check('MISSING', today=TODAY)
    assert result.status == Status.NON_COMPLIANT


def test_only_newest_snapshot_used_and_cash_and_remnants_excluded():
    old = ['11/13/2026', 'HLAL', 'OLD', 'Old Corp', 10, 5, 50, '']
    extra = [old]
    for ticker, name, shares, price, value, flag in [
        ('USD', 'US Dollar', 100, 1, 100, ''),
        ('CASH', 'Cash', 100, 1, 100, ''),
        ('MMKT', 'Example Corp', 100, 1, 100, 'Y'),
        ('XETF', 'Example ETF', 100, 1, 100, ''),
        ('2602335D', 'TPG Inc', 10, 1, 10, ''),
        ('ZERO', 'Old Corp', 10, 0, 0, ''),
        ('NAN', 'Invalid Corp', 10, 'NaN', 10, ''),
        ('INF', 'Invalid Corp', 10, 'inf', 10, ''),
    ]:
        extra.append(['11/16/2026', 'HLAL', ticker, name, shares, price, value, flag])
    holdings = parse_holdings(feed('HLAL', ['NEW', 'BRK B'], extra=extra), 'HLAL', today=TODAY)
    assert holdings.as_of == TODAY
    assert holdings.symbols == frozenset([*(f'ZZ{chr(65 + i)}' for i in range(26)), 'NEW', 'BRK.B'])


@pytest.mark.parametrize('symbol', ['brk.b', 'BRK-B', 'BRK B'])
def test_share_class_symbol_normalization(symbol):
    with patch('execution.etf_holdings.requests.get', side_effect=[response(feed('HLAL', ['BRK.B'])), response(feed('SPUS'))]):
        assert ETFHoldingsScreener().check(symbol, today=TODAY).compliant


@pytest.mark.parametrize('day', [TODAY - timedelta(days=8), TODAY + timedelta(days=1)])
def test_stale_or_future_published_dates_rejected(day):
    with pytest.raises(ValueError, match='stale or future'):
        parse_holdings(feed('HLAL', ['TEST'], day), 'HLAL', today=TODAY)


@pytest.mark.parametrize('text', ['<html>Access denied</html>', '', 'Date,StockTicker\n11/16/2026,TEST'])
def test_bad_schema_rejected(text):
    with pytest.raises(ValueError):
        parse_holdings(text, 'HLAL', today=TODAY)


def test_truncated_feed_and_wrong_account_rejected():
    text = feed('HLAL')
    with pytest.raises(ValueError, match='too few'):
        parse_holdings('\n'.join(text.splitlines()[:3]), 'HLAL', today=TODAY)
    with pytest.raises(ValueError, match='account'):
        parse_holdings(text, 'SPUS', today=TODAY)


def test_partial_outage_allows_match_but_not_unknown_stock():
    with patch('execution.etf_holdings.requests.get', side_effect=[requests.Timeout(), response(feed('SPUS', ['TEST']))]) as get:
        screener = ETFHoldingsScreener()
        assert screener.check('TEST', today=TODAY).compliant
        assert screener.check('OTHER', today=TODAY).status == Status.INSUFFICIENT_DATA
        assert get.call_count == 2  # Failed source is not hammered on every entry.


def test_daily_refresh_revokes_previous_membership():
    with patch(
        'execution.etf_holdings.requests.get',
        side_effect=[
            response(feed('HLAL', ['TEST'])),
            response(feed('SPUS')),
            response(feed('HLAL', day=TODAY + timedelta(days=1))),
            response(feed('SPUS', day=TODAY + timedelta(days=1))),
        ],
    ) as get:
        screener = ETFHoldingsScreener()
        assert screener.check('TEST', today=TODAY).compliant
        assert screener.check('TEST', today=TODAY).compliant
        assert get.call_count == 2
        assert not screener.check('TEST', today=TODAY + timedelta(days=1)).compliant
        assert get.call_count == 4


def test_failed_daily_refresh_does_not_reuse_cached_approval_and_retries():
    with (
        patch('execution.etf_holdings.time.monotonic', return_value=0) as clock,
        patch(
            'execution.etf_holdings.requests.get',
            side_effect=[
                response(feed('HLAL', ['TEST'])),
                response(feed('SPUS')),
                requests.Timeout(),
                requests.Timeout(),
                response(feed('HLAL', ['TEST'], TODAY + timedelta(days=1))),
                response(feed('SPUS', day=TODAY + timedelta(days=1))),
            ],
        ) as get,
    ):
        screener = ETFHoldingsScreener()
        assert screener.check('TEST', today=TODAY).compliant
        assert screener.check('TEST', today=TODAY + timedelta(days=1)).status == Status.INSUFFICIENT_DATA
        assert not screener.check('TEST', today=TODAY + timedelta(days=1)).compliant
        assert get.call_count == 4
        clock.return_value = 301
        assert screener.check('TEST', today=TODAY + timedelta(days=1)).compliant
        assert get.call_count == 6


@pytest.mark.parametrize('age', [0, 8, True, '7'])
def test_bad_age_config_rejected(age):
    with pytest.raises(ValueError):
        screener_from_config({'mode': 'etf_holdings', 'holdings_max_age_days': age})


def test_default_config_uses_etfs_and_unknown_mode_rejected():
    assert isinstance(screener_from_config({}), ETFHoldingsScreener)
    with pytest.raises(ValueError):
        screener_from_config({'mode': 'unknown'})


@pytest.mark.parametrize('symbol', [None, '', 'INVALID/SYMBOL'])
def test_invalid_symbols_do_not_fetch(symbol):
    with patch('execution.etf_holdings.requests.get') as get:
        assert ETFHoldingsScreener().check(symbol).status == Status.INSUFFICIENT_DATA
        get.assert_not_called()
