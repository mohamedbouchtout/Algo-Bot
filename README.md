# Algo-Bot: Automated Breakout & Retest Trading System

A sophisticated automated trading bot that scans S&P 500 and NASDAQ stocks for 200-day moving average breakout and retest patterns. Built with a modular architecture for reliability and maintainability.

## 🚀 Features

- **Automated Pattern Detection**: Scans for 200 MA breakout and retest patterns
- **Dual Strategy Support**: Long and short positions based on trend direction
- **Risk Management**: Position sizing, stop losses, take profit targets
- **Bracket Orders**: Automatic entry, stop loss, and take profit orders
- **Market Hours Detection**: Only trades during NYSE market hours
- **Git Integration**: Auto-updates from repository during off-hours
- **Environment Configuration**: Dev/Prod configs based on Git branch
- **Comprehensive Logging**: Detailed logs with timestamps
- **Paper Trading Ready**: Configured for IBKR paper trading

## 📁 Project Structure

```
Algo-Bot/
├── main.py                 # Application entry point
├── config/
│   ├── dev.json           # Development configuration
│   ├── prod.json          # Production configuration
│   └── trading_params.json # Trading strategy parameters
├── core/
│   ├── bot.py            # Main TradingBot orchestrator
│   ├── connection.py     # IBKR connection management
│   └── scheduler.py      # Market hours scheduling
├── data_fetch/
│   ├── stock_fetcher.py  # Stock list management
│   ├── historical_data.py # Historical price data
│   └── persistence.py    # Data persistence utilities
├── execution/
│   ├── order_manager.py  # Order placement and management
│   ├── position_manager.py # Position tracking
│   └── risk_manager.py   # Risk calculation and validation
├── strategy/
│   └── retest_200ma/
│       ├── indicators.py  # Technical indicators
│       ├── trend_detector.py # Pattern detection logic
│       └── validators.py  # Trend validation
├── utils/
│   ├── alerts.py         # Alert system
│   ├── git_manager.py    # Git operations
│   ├── logger.py         # Logging utilities
│   └── metrics.py        # Performance metrics
├── tests/                # Unit tests
├── data/                 # Runtime data storage
│   ├── logs/            # Log files
│   └── performance/     # Performance data
└── requirements.txt      # Python dependencies
```

## 📈 Strategy Overview

The bot implements a 200-day moving average breakout and retest strategy:

### Long Setup
1. **Breakout**: Price crosses above 200 MA with high volume
2. **Retest**: Price returns to test 200 MA from above
3. **Bounce**: Price bounces upward off the MA
4. **Entry**: Long position with 2:1 risk/reward ratio

### Short Setup
1. **Breakdown**: Price crosses below 200 MA with high volume
2. **Retest**: Price returns to test 200 MA from below
3. **Bounce**: Price bounces downward off the MA
4. **Entry**: Short position with 2:1 risk/reward ratio

### Key Parameters
- **MA Period**: 200 days
- **Lookback**: 250 trading days of historical data
- **Risk/Reward**: 2:1 minimum
- **Volume Filter**: Breakout must be 2x average volume
- **Retest Distance**: Within 0.5% of MA
- **Max Days Since Retest**: 3 days

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Interactive Brokers account (paper trading recommended)
- TWS or IB Gateway installed

### 1. Clone Repository
```bash
git clone https://github.com/mohamedbouchtout/Algo-Bot.git
cd Algo-Bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Interactive Brokers

1. **Download TWS or IB Gateway**
   - Visit interactivebrokers.com
   - Download Trader Workstation (TWS) or IB Gateway

2. **Configure API Access**
   - Launch TWS/IB Gateway
   - File → Global Configuration → API → Settings
   - ✅ Enable ActiveX and Socket Clients
   - Socket port: 9000 (paper trading)
   - ✅ Allow connections from localhost only
   - ✅ Read-only API (for safety)

3. **Paper Trading Account**
   - Create paper trading account at IBKR
   - Fund with virtual currency
   - Use paper trading credentials

## ⚙️ Configuration

The bot uses JSON configuration files in the `config/` directory:

### Environment Selection
- **Development**: `config/dev.json` (when on non-production branches)
- **Production**: `config/prod.json` (when on `bot/production` branch)

### Trading Parameters (`config/trading_params.json`)
```json
{
  "strategy_retest_200ma": {
    "ma_period": 200,
    "risk_reward_ratio": 2.0,
    "lookback_days": 250,
    "min_breakout_volume": 2.0,
    "retest_distance": 0.005
  },
  "risk_management": {
    "risk_per_trade_pct": 0.01,
    "max_investment_pct": 0.70,
    "max_positions": 10
  },
  "timing": {
    "scan_interval": 1200,
    "market_check_interval": 900
  }
}
```

### IBKR Configuration
```json
{
  "ib": {
    "host": "127.0.0.1",
    "port": 9000,
    "client_id": 1
  }
}
```

## 🚀 Usage

### Start the Bot
```bash
python main.py
```

### What the Bot Does
- ✅ Connects to IBKR TWS/Gateway
- ✅ Loads stock list (S&P 500 + NASDAQ-100)
- ✅ Runs continuously during market hours
- ✅ Scans stocks every 20 minutes for signals
- ✅ Places bracket orders when patterns detected
- ✅ Monitors existing positions
- ✅ Auto-commits changes to Git
- ✅ Logs all activity

### Monitor Activity
The bot creates timestamped log files in `data/logs/`:
```
data/logs/trading_bot_3-22-2026_14-30.log
```

Example log output:
```
2026-03-22 14:30:00 - INFO - Starting trading bot...
2026-03-22 14:30:01 - INFO - Connected to IB at 127.0.0.1:9000
2026-03-22 14:30:02 - INFO - Scanning 650 stocks...
2026-03-22 14:32:15 - INFO - Signal found: LONG AAPL @ $225.50
2026-03-22 14:32:16 - INFO - Entered position: AAPL LONG, 40 shares @ $225.50
```

## 🛡️ Risk Management

### Built-in Safeguards
- ✅ **Position Sizing**: 1% account risk per trade
- ✅ **Max Investment**: 70% of account can be invested
- ✅ **Max Positions**: 10 concurrent positions
- ✅ **Stop Losses**: Automatic on all trades
- ✅ **Take Profits**: 2:1 reward targets
- ✅ **Bracket Orders**: Entry + Stop + Target in one order
- ✅ **One Position Per Stock**: Prevents overexposure

### Going Live (Production)
1. Switch to production branch: `git checkout bot/production`
2. Update `config/prod.json` with live account settings
3. Change IBKR port to 7496 (live trading)
4. Reduce risk to 0.5% per trade
5. Start with small position sizes
6. Monitor closely for first month

## 🧪 Testing

### Run Tests
```bash
python -m pytest tests/
```

### Test Coverage
- `test_position_manager.py`: Position tracking logic
- `test_retest_200ma.py`: Strategy pattern detection
- `test_risk_manager.py`: Risk calculation validation

### Paper Trading Checklist
- [ ] Run for 30+ trading days
- [ ] Track win rate (>50% target)
- [ ] Monitor max drawdown (<10%)
- [ ] Verify signal quality manually
- [ ] Test during volatile markets
- [ ] Check error handling

## 🔧 Troubleshooting

### Connection Issues
**"Connection Refused"**
- Ensure TWS/IB Gateway is running
- Check API settings are enabled
- Verify correct port (9000 for paper, 7496 for live)

**"No Market Data"**
- Accept market data agreements in IBKR Account Management
- Check if markets are open (9:30 AM - 4:00 PM EST weekdays)

### Trading Issues
**"Position Size Too Small"**
- Account balance too low for risk parameters
- Increase account size or reduce `risk_per_trade_pct`

**"No Signals Found"**
- Pattern is rare - may take days
- Check logs for scanning activity
- Verify historical data loading

### Configuration Issues
**"Failed to load configuration"**
- Check JSON syntax in config files
- Ensure config files exist in `config/` directory
- Verify Git branch detection for environment selection

## 📊 Performance Monitoring

### Key Metrics Tracked
- Win Rate: Percentage of profitable trades
- Average R:R: Risk-reward ratio achieved
- Max Drawdown: Peak-to-valley decline
- Sharpe Ratio: Risk-adjusted returns
- Position Holding Time: Average duration

### Log Analysis
```bash
# View recent activity
tail -f data/logs/trading_bot_*.log

# Count signals found
grep "Signal found" data/logs/trading_bot_*.log | wc -l

# Check position entries
grep "Entered position" data/logs/trading_bot_*.log
```

## 🚀 Deployment

### Local Machine
- Keep computer running during market hours
- Bot automatically pauses when market closed

### Cloud Server (Recommended)
```bash
# On VPS (AWS, DigitalOcean, etc.)
nohup python main.py > bot.log 2>&1 &
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## 📝 Development

### Adding New Strategies
1. Create new module in `strategy/`
2. Implement pattern detection logic
3. Add parameters to `trading_params.json`
4. Update `order_manager.py` to call new strategy

### Code Quality
- Follow PEP 8 style guidelines
- Add type hints to function signatures
- Write comprehensive unit tests
- Use logging instead of print statements

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## ⚠️ Important Notes

### Limitations
- **Delayed Data**: Uses free 15-minute delayed data
- **Scanning Time**: ~5 minutes for 650 stocks
- **Market Orders**: May experience slippage
- **IBKR Dependency**: Requires active IBKR connection

### Legal Disclaimer
This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly before using real money.

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Open an issue on GitHub
4. Ensure all prerequisites are met
- Market closes
- Bot pauses scanning
- Monitors existing positions

**Continues running 24/7...**

## Support

For issues:
1. Check `trading_bot_<month>-<day>-<year>_<hour>-<minute>.log` for errors
2. Verify TWS is running and API enabled
3. Ensure markets are open (for live data)
4. Review IB API documentation: https://interactivebrokers.github.io/

## Disclaimer

This is for educational purposes. Trading involves risk. Test thoroughly with paper trading before using real money. Past performance doesn't guarantee future results.