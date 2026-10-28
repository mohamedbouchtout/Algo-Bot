"""Evidence-based equity screening. See docs/sharia-screening.md for policy details."""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Protocol

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
POLICY = 'ftse-yasaar-numerical-v4.6-2026-02'


class Status(str, Enum):
    COMPLIANT = 'compliant'
    NON_COMPLIANT = 'non_compliant'
    INSUFFICIENT_DATA = 'insufficient_data'


@dataclass(frozen=True)
class FinancialSnapshot:
    """Amounts use one currency/unit; income covers the year ending as_of.

    non_compliant_revenue excludes interest_income to prevent double counting.
    Reviewed business evidence covers all prohibited activities and subsidiaries.
    """

    symbol: str
    as_of: date
    source: str
    currency: str
    total_assets: object = None
    total_debt: object = None
    cash: object = None
    interest_bearing_investments: object = None
    receivables: object = None
    revenue: object = None
    interest_income: object = None
    non_compliant_revenue: object = None
    business_reviewed: bool = False
    industry: str = ''


@dataclass(frozen=True)
class ScreeningResult:
    symbol: str
    status: Status
    reasons: tuple[str, ...]
    as_of: date | None = None
    source: str = ''
    policy: str = POLICY
    ratios: dict[str, float] = field(default_factory=dict)

    @property
    def compliant(self) -> bool:
        return self.status == Status.COMPLIANT


def normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.^=-]{0,19}', symbol.strip()):
        raise ValueError('Invalid stock symbol')
    return symbol.strip().upper()


def _amount(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
        return number if number.is_finite() and number >= 0 else None
    except (InvalidOperation, ValueError):
        return None


def screen_snapshot(snapshot: FinancialSnapshot, *, today: date | None = None, max_age_days: int = 450) -> ScreeningResult:
    """Apply published numerical screens without index turnover buffers.

    No keyword or missing-value default can establish a passing result.
    """
    today = today or date.today()
    problems = []
    if not isinstance(snapshot.as_of, date) or not 0 <= (today - snapshot.as_of).days <= max_age_days:
        problems.append('Financial evidence is stale, future-dated, or has an invalid date')
    if not snapshot.source.strip() or not re.fullmatch(r'[A-Z]{3}', snapshot.currency):
        problems.append('Source and ISO currency are required')
    names = (
        'total_assets',
        'total_debt',
        'cash',
        'interest_bearing_investments',
        'receivables',
        'revenue',
        'interest_income',
        'non_compliant_revenue',
    )
    amounts = {name: _amount(getattr(snapshot, name)) for name in names}
    for name, amount in amounts.items():
        if amount is None or (name in ('total_assets', 'revenue') and amount == 0):
            problems.append(f'Missing or invalid {name}')
    if snapshot.business_reviewed is not True:
        problems.append('Complete prohibited-activity revenue review is required')
    if problems:
        return ScreeningResult(snapshot.symbol, Status.INSUFFICIENT_DATA, tuple(problems), snapshot.as_of, snapshot.source)
    a = {name: Decimal(str(getattr(snapshot, name))) for name in names}
    if a['cash'] + a['receivables'] + a['interest_bearing_investments'] > a['total_assets'] or a['non_compliant_revenue'] > a['revenue']:
        return ScreeningResult(snapshot.symbol, Status.INSUFFICIENT_DATA, ('Inconsistent financial amounts',), snapshot.as_of, snapshot.source)
    ratios = {
        'debt_to_assets': a['total_debt'] / a['total_assets'],
        'cash_and_interest_to_assets': (a['cash'] + a['interest_bearing_investments']) / a['total_assets'],
        'cash_and_receivables_to_assets': (a['cash'] + a['receivables']) / a['total_assets'],
        'non_permissible_income_to_revenue': (a['interest_income'] + a['non_compliant_revenue']) / a['revenue'],
    }
    limits = (Decimal('0.33333'), Decimal('0.33333'), Decimal('0.50'), Decimal('0.05'))
    failures = []
    for (name, ratio), limit in zip(ratios.items(), limits):
        if ratio > limit or (name != 'non_permissible_income_to_revenue' and ratio == limit):
            failures.append(f'{name} fails {"<=" if name == "non_permissible_income_to_revenue" else "<"} {limit}')
    return ScreeningResult(
        snapshot.symbol,
        Status.NON_COMPLIANT if failures else Status.COMPLIANT,
        tuple(failures),
        snapshot.as_of,
        snapshot.source,
        ratios={k: float(v) for k, v in ratios.items()},
    )


class SnapshotProvider(Protocol):
    def fetch(self, symbol: str) -> FinancialSnapshot: ...


class ReviewedFileProvider:
    """Reload reviewed evidence every check, making revocations effective immediately."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch(self, symbol: str) -> FinancialSnapshot:
        with self.path.open(encoding='utf-8') as handle:
            records = json.load(handle, parse_float=Decimal)
        record = dict(records[symbol])
        record['as_of'] = date.fromisoformat(record['as_of'])
        return FinancialSnapshot(symbol=symbol, **record)


class YahooSnapshotProvider:
    """Partial annual evidence: Yahoo cannot establish a complete business review.

    Never infer zero prohibited revenue or investments from absent fields. Never
    mix fiscal dates or substitute quote ratios/spot market capitalization.
    """

    def fetch(self, symbol: str) -> FinancialSnapshot:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        balance = ticker.balance_sheet
        income = ticker.income_stmt
        if balance.empty or income.empty:
            raise ValueError('Annual financial statements unavailable')
        latest = max(balance.columns)
        if latest not in income.columns:
            raise ValueError('Latest balance sheet and income statement periods differ')

        def value(frame: pd.DataFrame, *labels: str) -> object:
            for label in labels:
                if label in frame.index:
                    result = frame.at[label, latest]
                    if pd.notna(result):
                        return result
            return None

        return FinancialSnapshot(
            symbol=symbol,
            as_of=pd.Timestamp(latest).date(),
            source='Yahoo Finance annual statements',
            currency=info.get('financialCurrency', ''),
            industry=info.get('industry', ''),
            total_assets=value(balance, 'Total Assets'),
            total_debt=value(balance, 'Total Debt'),
            cash=value(balance, 'Cash And Cash Equivalents'),
            receivables=value(balance, 'Accounts Receivable', 'Net Receivables'),
            revenue=value(income, 'Total Revenue'),
            interest_income=value(income, 'Interest Income'),
        )


class ShariaScreener:
    def __init__(self, provider: SnapshotProvider | None = None, *, max_age_days: int = 450):
        if isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or not 1 <= max_age_days <= 450:
            raise ValueError('max_age_days must be an integer between 1 and 450')
        self.provider = provider if provider is not None else YahooSnapshotProvider()
        self.max_age_days = max_age_days

    def check(self, symbol: str, *, today: date | None = None) -> ScreeningResult:
        try:
            symbol = normalize_symbol(symbol)
            snapshot = self.provider.fetch(symbol)
            if snapshot.symbol != symbol:
                raise ValueError('Evidence symbol does not match requested symbol')
            result = screen_snapshot(snapshot, today=today, max_age_days=self.max_age_days)
        except Exception as exc:
            logger.warning('Sharia evidence unavailable for %s: %s', symbol, exc)
            result = ScreeningResult(symbol, Status.INSUFFICIENT_DATA, ('Evidence unavailable or malformed',))
        logger.info(
            'Sharia screen %s: %s (%s), source=%s, as_of=%s, reasons=%s',
            symbol,
            result.status.value,
            result.policy,
            result.source,
            result.as_of,
            result.reasons,
        )
        return result


def screener_from_config(config: dict) -> ShariaScreener:
    if config.get('methodology', POLICY) != POLICY:
        raise ValueError('Unsupported Sharia screening methodology')
    path = config.get('evidence_file')
    return ShariaScreener(ReviewedFileProvider(path) if path else None, max_age_days=config.get('max_age_days', 450))


def is_sharia_compliant(stock_symbol: str) -> bool:
    """Compatibility API: only a fully evidenced pass returns True."""
    return ShariaScreener().check(stock_symbol).compliant
