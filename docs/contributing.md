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

This runs linting, unit tests, and integration tests across Python 3.11–3.14. You can also run individual sessions:

```bash
nox -s lint              # ruff linting + format check
nox -s fix               # auto-fix lint errors and reformat code
nox -s test              # unit tests on all Python versions (3.11–3.14)
nox -s test-3.12         # unit tests on a specific Python version
nox -s integration       # integration tests on all Python versions
nox -s integration-3.12  # integration tests on a specific Python version
nox -s typecheck         # mypy type checking
```

Run `nox -s fix` before pushing to auto-format your code and fix lint errors.

### Test Structure

Tests are organized into subdirectories under `tests/`:

```text
tests/
├── conftest.py              # Shared fixtures (PARAMS, CONFIG, synthetic data helpers)
├── unit/                    # Fast unit tests — no network or IB required
│   ├── test_risk_manager.py # Position sizing, trade validation, stop loss
│   ├── test_cnn.py          # CNN model, early stopping
│   ├── test_lstm.py         # LSTM network and trainer
│   ├── test_walk_forward.py # Walk-forward cross-validation splits
│   ├── test_features.py     # Volatility-adjusted labels, market features
│   └── test_retrain_trigger.py  # Regime shift and accuracy checks
├── integration/             # End-to-end pipeline tests (some need network)
│   ├── test_data_fetch.py       # yfinance data fetching
│   ├── test_feature_pipeline.py # Full feature extraction chain
│   ├── test_ai_pipeline.py      # Train + predict + walk-forward validation
│   ├── test_strategy_200ma.py   # 200-MA pattern detection
│   ├── test_signals.py          # Signal construction + risk sizing flow
│   ├── test_retrain_trigger.py  # Retrain trigger with live data
│   ├── test_scheduler.py        # Market hours checks
│   └── test_end_to_end.py       # Full real-data pipeline
└── legacy/                  # Old scripts (not collected by pytest)
```

**Unit tests** (`tests/unit/`) run without network access or IB and are selected by default with `nox -s test`.

**Integration tests** (`tests/integration/`) exercise real pipelines end-to-end. Tests that hit yfinance are skipped gracefully when offline. Run them with `nox -s integration`.

### Writing Tests

- **Unit tests**: Place in `tests/unit/`. No special marker needed.
- **Integration tests**: Place in `tests/integration/` and mark every class with `@pytest.mark.integration`.
- Use `make_synthetic_bars()` and `PARAMS`/`CONFIG` from `tests/conftest.py` for shared test data.
- Tests that need network access should use the `@requires_network` decorator from `tests/conftest.py`.

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
