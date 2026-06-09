# Algo-Bot

[![CI](https://github.com/mohamedbouchtout/Algo-Bot/actions/workflows/ci.yml/badge.svg)](https://github.com/mohamedbouchtout/Algo-Bot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

An automated trading bot that scans S&P 500 and NASDAQ stocks for breakout and retest patterns using a 200-day moving average strategy, with an AI analysis pipeline (LSTM/CNN) for signal classification.

Built for Interactive Brokers (IBKR) with bracket orders, risk management, email alerts, and auto-updates.

## How It Works

The bot runs a continuous loop during NYSE market hours:

1. **Scans** 500+ stocks for 200 MA breakout/retest patterns
2. **Validates** signals with volume, trend direction, and bounce confirmation
3. **Classifies** with an LSTM model trained on price, volume, indicator, and market regime features
4. **Sizes positions** based on per-trade risk and account equity
5. **Places bracket orders** (entry + stop loss + take profit) through IBKR
6. **Monitors** open positions and sends email alerts on entry/exit

Outside market hours the bot sleeps, checks for code updates, and retrains AI models on weekends.

## Quick Start

```bash
git clone https://github.com/mohamedbouchtout/Algo-Bot.git
cd Algo-Bot
pip install -r requirements.txt
python main.py
```

Requires Python 3.10+ and TWS or IB Gateway running with API enabled. See [Installation](docs/installation.md) for full setup instructions.

## Project Structure

```
Algo-Bot/
├── main.py                  # Entry point
├── config/                  # Connection + strategy configs (single config.json)
├── core/                    # Bot orchestrator, IB connection, scheduler
├── data_fetch/              # Stock list + historical data (via yfinance)
├── execution/               # Order placement, position tracking, risk management
├── strategy/
│   ├── retest_200ma/        # 200 MA breakout/retest pattern detection
│   └── ai_analysis/         # LSTM/CNN feature pipeline and prediction
├── ai_modules/              # LSTM and CNN model implementations (PyTorch)
├── utils/                   # Email alerts, git manager, logging
├── tests/                   # Test suite
└── data/                    # Runtime logs, positions, stock lists (auto-created)
```

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](docs/installation.md) | Prerequisites, setup, IB configuration, Docker |
| [Configuration](docs/configuration.md) | Config files, trading parameters, environment variables |
| [Strategies](docs/strategies.md) | 200 MA strategy details and AI analysis pipeline |
| [Risk Management](docs/risk-management.md) | Position sizing, safeguards, going live |
| [Deployment](docs/deployment.md) | Running locally, on a server, or in Docker |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and how to fix them |
| [Contributing](docs/contributing.md) | How to add features, code standards, PR workflow |

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly with paper trading before using real money.

## License

MIT License - see [LICENSE](LICENSE) for details.
