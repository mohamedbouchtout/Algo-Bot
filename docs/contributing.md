# Contributing

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/Algo-Bot.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a feature branch: `git checkout -b feature/your-feature`

## Code Standards

- Code is linted and formatted with [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`)
- Add type hints to function signatures
- Use `logging` instead of `print` statements
- Use the project's existing logger setup: `logger = logging.getLogger()`

## Adding a New Strategy

1. Create a new module in `strategy/` (e.g., `strategy/your_strategy/`)
2. Implement pattern detection logic that returns a signal dict:
   ```python
   {
       'strategy_type': 'your_strategy',
       'type': 'LONG' or 'SHORT',
       'symbol': 'AAPL',
       'entry': 150.00,
       'stop': 147.00,
       'target': 156.00,
       'risk': 3.00,
       'reward': 6.00,
   }
   ```
3. Add any new parameters to `config/trading_params.json`
4. Integrate with `order_manager.py` to call your strategy during scans
5. Add persistence support in `position_manager.py` if your signal has custom fields

## Running Checks

Run all checks locally before pushing (same checks that CI runs):

```bash
nox
```

This runs linting and unit tests. You can also run individual sessions:

```bash
nox -s lint          # ruff linting + format check
nox -s test          # pytest unit tests
nox -s typecheck     # mypy type checking
nox -s fix           # auto-fix lint errors and reformat code
```

Run `nox -s fix` before pushing to auto-format your code and fix lint errors.

### Integration Tests

The full integration test suite requires TWS or IB Gateway running:

```bash
python -c "from tests.run_tests import RunTests; RunTests().run()"
```

### Test Files

- `test_risk_manager.py` — Position sizing and risk calculation (unit tests, no IB needed)
- `test_ai_enhancements.py` — LSTM, CNN, walk-forward, volatility labels, market features (unit tests, no IB needed)
- `test_retest_200ma.py` — Strategy pattern detection (integration, needs IB)
- `test_ai_analysis.py` — AI pipeline training and prediction (integration, needs IB)
- `test_ai_backtest.py` — Per-sector walk-forward AI performance backtest (integration, needs IB)

Test logs are written to `data/test_logs/`.

## Project Architecture

```
main.py
  -> core/bot.py (TradingBot)
       -> core/connection.py      IB connection lifecycle
       -> core/scheduler.py       Market hours detection
       -> data_fetch/             Stock lists + historical data
       -> execution/              Order placement, positions, risk
       -> strategy/               Signal generation
       -> utils/                  Alerts, git, logging
```

Key design decisions:
- **One IB instance** shared across all modules via dependency injection
- **Config/params separation**: Connection settings in `config.json`, strategy parameters in `trading_params.json`
- **Per-sector AI models**: Each sector gets its own LSTM (or CNN) to avoid cross-sector noise
- **Bracket orders**: Every trade is a parent + stop loss + take profit, submitted atomically

## Pull Request Workflow

1. Run `nox` to verify linting and tests pass
3. Commit with a clear message describing what and why
4. Push to your fork
5. Open a pull request against `main`
6. Describe what changed, why, and how to test it
