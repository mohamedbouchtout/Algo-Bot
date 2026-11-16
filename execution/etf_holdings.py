"""Automatic stock eligibility from the official HLAL and SPUS holdings feeds."""

import csv
import io
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime

import requests

from execution.sharia_compliance import ScreeningResult, Status, normalize_symbol

logger = logging.getLogger(__name__)
ETF_POLICY = 'hlal-spus-holdings-union-v1'
HOLDINGS_URLS = {
    # CSV export of the Holdings sheet linked by https://www.wahed.com/hlal.
    'HLAL': 'https://docs.google.com/spreadsheets/d/1UC1Bk67bGuYsos_i8y_HQpNoHpVHAvqf71MbgrafJOQ/export?format=csv&gid=0',
    'SPUS': 'https://www.sp-funds.com/wp-content/uploads/data/TidalFG_Holdings_SPUS.csv',
}


@dataclass(frozen=True)
class Holdings:
    as_of: date
    symbols: frozenset[str]


def stock_symbol(value: str) -> str:
    """Treat broker, Yahoo and fund share-class separators consistently."""
    return normalize_symbol(value).replace('-', '.').replace(' ', '.')


def parse_holdings(text: str, fund: str, *, today: date, max_age_days: int = 7) -> Holdings:
    """Read only the newest dated snapshot; never union historical holdings."""
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    required = {'Date', 'Account', 'StockTicker', 'SecurityName', 'Shares', 'Price', 'MarketValue'}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError('Unexpected holdings CSV columns')
    dated_rows = []
    for row in reader:
        if not any(row.values()):
            continue
        if row['Account'].strip().upper() != fund:
            raise ValueError('Holdings account does not match fund')
        as_of = datetime.strptime(row['Date'].strip(), '%m/%d/%Y').date()
        dated_rows.append((as_of, row))
    if not dated_rows:
        raise ValueError('Empty holdings feed')
    latest = max(day for day, _ in dated_rows)
    if not 0 <= (today - latest).days <= max_age_days:
        raise ValueError('Holdings date is stale or future-dated')
    symbols = set()
    for day, row in dated_rows:
        if day != latest:
            continue
        ticker = row['StockTicker'].strip().upper()
        name = row['SecurityName'].strip().upper()
        # Ignore cash, money-market instruments, identifiers/CVRs and zero-value remnants.
        if row.get('MoneyMarketFlag', '').strip().upper() in {'Y', 'YES', 'TRUE', '1'}:
            continue
        if ticker in {'CASH', 'USD', 'HLAL', 'SPUS'} or re.search(r'\b(CASH|MONEY MARKET|TREASURY|FUND|ETF)\b', name):
            continue
        if not re.fullmatch(r'[A-Z]{1,6}(?:[.\- ][A-Z])?', ticker):
            continue
        amounts = [float(row[key].replace(',', '')) for key in ('Shares', 'Price', 'MarketValue')]
        if not all(0 < amount < float('inf') for amount in amounts):
            continue
        symbols.add(stock_symbol(ticker.replace(' ', '.')))
    # These diversified funds hold far more than 25 stocks. Reject empty/truncated exports.
    if len(symbols) < 25:
        raise ValueError('Holdings feed contains too few stocks')
    return Holdings(latest, frozenset(symbols))


class ETFHoldingsScreener:
    """Allow membership in either fresh fund; refresh daily and retry failures in 5 minutes."""

    def __init__(self, *, max_age_days: int = 7):
        if isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or not 1 <= max_age_days <= 7:
            raise ValueError('holdings_max_age_days must be an integer between 1 and 7')
        self.max_age_days = max_age_days
        self._holdings: dict[str, Holdings] = {}
        self._fetched_on: dict[str, date] = {}
        self._retry_at: dict[str, float] = {}

    def _refresh(self, today: date):
        for fund, url in HOLDINGS_URLS.items():
            if self._fetched_on.get(fund) == today:
                continue
            if time.monotonic() < self._retry_at.get(fund, 0):
                continue
            # A previous day's cache must not survive a failed daily refresh.
            self._holdings.pop(fund, None)
            try:
                response = requests.get(url, timeout=(5, 20), headers={'User-Agent': 'Algo-Bot ETF holdings screener'})
                response.raise_for_status()
                holdings = parse_holdings(response.text, fund, today=today, max_age_days=self.max_age_days)
                self._holdings[fund] = holdings
                self._fetched_on[fund] = today
                self._retry_at.pop(fund, None)
                logger.info('ETF holdings refreshed: %s as_of=%s stocks=%d source=%s', fund, holdings.as_of, len(holdings.symbols), url)
            except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError, csv.Error) as exc:
                self._retry_at[fund] = time.monotonic() + 300
                logger.warning('ETF holdings unavailable for %s: %s', fund, exc)

    def check(self, symbol: str, *, today: date | None = None) -> ScreeningResult:
        today = today or date.today()
        try:
            symbol = stock_symbol(symbol.strip().replace(' ', '.'))
        except (ValueError, AttributeError):
            return ScreeningResult(str(symbol), Status.INSUFFICIENT_DATA, ('Invalid stock symbol',), policy=ETF_POLICY)
        self._refresh(today)
        fresh = {fund: h for fund, h in self._holdings.items() if 0 <= (today - h.as_of).days <= self.max_age_days}
        matches = [fund for fund, h in fresh.items() if symbol in h.symbols]
        if matches:
            status = Status.COMPLIANT
            reasons = (f'Listed in {", ".join(matches)} stock holdings',)
            as_of = min(fresh[fund].as_of for fund in matches)
        else:
            # Absence is conclusive only when both lists are available.
            status = Status.NON_COMPLIANT if len(fresh) == len(HOLDINGS_URLS) else Status.INSUFFICIENT_DATA
            reasons = (
                ('Not in either ETF stock holdings list',)
                if status == Status.NON_COMPLIANT
                else ('No membership match; one or more ETF feeds unavailable',)
            )
            as_of = None
        result = ScreeningResult(symbol, status, reasons, as_of, '; '.join(f'{fund}: {HOLDINGS_URLS[fund]}' for fund in matches or fresh), ETF_POLICY)
        logger.info('Sharia ETF screen %s: %s (%s), source=%s, as_of=%s, reasons=%s', symbol, status.value, ETF_POLICY, result.source, as_of, reasons)
        return result
