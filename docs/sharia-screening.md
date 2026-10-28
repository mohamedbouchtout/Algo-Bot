# Sharia equity screening

Enable `sharia_compliance.enabled` in `config/trading_params.json`. Every new
entry is screened at sizing and again before broker access. Only a `LONG` with a
`compliant` result proceeds. AI shorts, technical shorts, and direct order calls
are blocked. Existing-position exits and protective bracket orders remain available.
This screens equity selection; it does not validate the broker account's interest,
margin arrangements, settlement, or the rest of an investor's obligations.

## Policy

`ftse-yasaar-numerical-v4.6-2026-02` implements the published numerical screens
in [FTSE Yasaar Ground Rules v4.6, February 2026, sections 4.3.1–4.3.2](https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/ftse-yasaar-global-equity-shariah-index-series-ground-rules.pdf).
The [official overview](https://www.lseg.com/en/ftse-russell/indices/global-shariah)
also describes these screens. Other methodologies differ; this is a named policy,
not an assertion that every Sharia board reaches the same decision.

| Test | Required value |
| --- | --- |
| Total debt / total assets | < 33.333% |
| (Cash + interest-bearing investments) / total assets | < 33.333% |
| (Cash + accounts receivable) / total assets | < 50% |
| (Interest income + non-compliant activity revenue) / annual revenue | <= 5% |

Income from prohibited activities is aggregated across all business segments and
subsidiaries. This includes conventional finance/insurance, alcohol/intoxicants,
pork/non-halal food, gambling/cinema/music/pornography/hotels, tobacco/vapes/cannabis,
and weapons/defence manufacturing. The published activity list is not exhaustive;
a qualified review must cover all relevant activities, rather than search a company
description for keywords. Interest income must be gross income, not net interest
expense, and must not be counted again as non-compliant activity revenue.

Decimal arithmetic enforces the exact published boundaries. Zero debt is allowed;
zero assets/revenue, negative values, NaN, infinity, booleans and absent amounts are
invalid. The debt-to-equity quote field and spot market cap are never used.

This is a point-in-time equity screen. It does not reproduce official index
membership, the index's two-quarter turnover buffers, or Yasaar's proprietary
research/board approval. A passing numerical result is not a certification or fatwa.

## Data and results

`ShariaScreener.check(symbol)` returns `ScreeningResult` with status, reasons,
ratios, policy identifier, statement date and source. Status is one of:

- `compliant`: complete reviewed evidence passes every test.
- `non_compliant`: complete valid evidence fails one or more tests.
- `insufficient_data`: missing, inconsistent, stale, malformed or unavailable evidence.

Both latter statuses block new entries. Evidence quality is checked before ratios;
a partial failure is not represented as a complete compliance assessment. Results
are logged through Python logging, never printed. Symbols are normalized and
matched to the evidence record. No passing results are cached.

The default Yahoo provider retrieves annual balance sheets and income statements
for the latest balance-sheet fiscal date. It rejects mismatched dates, does not
fall back to an older complete period, and does not mix quote currencies with
statement currencies. **Yahoo does not provide a complete prohibited-revenue review
or a reliable interest-bearing investment breakdown. It therefore cannot produce
a passing result on its own.** An industry label cannot establish business
compliance; Islamic financial institutions and diversified companies especially
require actual revenue evidence.

Provide reviewed evidence through `evidence_file`, or inject a `SnapshotProvider`
into `ShariaScreener` to connect a research/data service. Providers must return a
complete `FinancialSnapshot` using the same accounting currency and unit for all
amounts, balance-sheet amounts at `as_of`, and income for the full year ending on
that date. Missing is `null`, never an inferred zero. Debt is total debt; investments
exclude the cash already in `cash`. Receivables exclude cash and investments.

`max_age_days` defaults to 450 (annual reporting plus filing lag); it can be reduced
to any integer from 1 to 450. Future-dated evidence is rejected. Maintain evidence
when new filings or business changes occur; reloads are immediate, but the program
cannot independently establish whether a reviewer omitted a new filing or activity.
No example data or presumed-compliant real tickers are shipped as trusted evidence.

## Reviewed evidence file

Set configuration as follows (paths resolve from the bot's working directory):

```json
"sharia_compliance": {
    "enabled": true,
    "methodology": "ftse-yasaar-numerical-v4.6-2026-02",
    "evidence_file": "data/sharia_evidence.json",
    "max_age_days": 450
}
```

Create the evidence file as a symbol-keyed JSON object. The following **synthetic**
record illustrates the schema; replace it with reviewed filings and actual source
references. Decimal strings preserve all source precision. `business_reviewed`
means the revenue review above is complete, not that someone approved the ticker.

```json
{
    "EXAMPLE": {
        "as_of": "2026-06-30",
        "source": "Annual filing URL, financial note/segment references; reviewer and review date",
        "currency": "USD",
        "total_assets": "1000",
        "total_debt": "100",
        "cash": "100",
        "interest_bearing_investments": "50",
        "receivables": "100",
        "revenue": "1000",
        "interest_income": "10",
        "non_compliant_revenue": "20",
        "business_reviewed": true
    }
}
```

An absent symbol, invalid JSON, unexpected field, or unreadable file fails closed.
The file is reloaded on every check, including the broker boundary, so revised
research can revoke approval between sizing and submission. Write updates atomically
to avoid transient malformed reads. The file is a trusted research input: the bot
validates structure and arithmetic, not the truth or completeness of the review.

For programmatic use:

```python
from execution.sharia_compliance import ReviewedFileProvider, ShariaScreener

screener = ShariaScreener(ReviewedFileProvider('data/sharia_evidence.json'))
result = screener.check('EXAMPLE')
print(result.status.value, result.reasons, result.ratios)
```

The original `is_sharia_compliant(symbol)` boolean API is retained and returns true
only for a full pass. Its default Yahoo data cannot establish one; use the configured
screener for reviewed evidence. Disabled trading mode makes no screening requests.

## Tests

```bash
python -m pytest tests/unit/test_sharia_compliance.py tests/integration/test_sharia_order_flow.py -q
```

The tests use fixed snapshots and mocked Yahoo/IB boundaries. They cover each ratio,
exact limits, aggregation of interest/activity income, missing/invalid/stale data,
provider errors, fiscal-period selection, evidence reload/revocation, and both AI
and technical signal paths. No live broker or known-ticker assumptions are required.
