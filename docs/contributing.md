# Contributing

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/Algo-Bot.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a feature branch: `git checkout -b feature/your-feature`

## Code Standards

- Follow PEP 8 style guidelines
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

## Running Tests

```bash
python tests_main.py
```

Test files are in `tests/`:
- `test_retest_200ma.py` — Strategy pattern detection
- `test_ai_analysis.py` — AI pipeline data flow
- `test_position_manager.py` — Position tracking logic
- `test_risk_manager.py` — Risk calculation validation

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
- **Per-sector AI models**: Each sector gets its own RBM + CNN to avoid cross-sector noise
- **Bracket orders**: Every trade is a parent + stop loss + take profit, submitted atomically

## Pull Request Workflow

1. Make sure all tests pass: `python tests_main.py`
2. Verify syntax: `python -m py_compile <your_file>.py`
3. Commit with a clear message describing what and why
4. Push to your fork
5. Open a pull request against `main`
6. Describe what changed, why, and how to test it
