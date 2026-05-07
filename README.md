# Algo-Bot: Automated Breakout & Retest Trading System

A sophisticated automated trading bot that scans S&P 500 and NASDAQ stocks for 200-day moving average breakout and retest patterns. Built with a modular architecture for reliability and maintainability.

## 🚀 Features

- **Automated Pattern Detection**: Scans for 200 MA breakout and retest patterns
- **Dual Strategy Support**: Long and short positions based on trend direction
- **AI Analysis Pipeline**: Optional RBM + CNN pipeline that learns features from
  stock price, volume and technical-indicator windows and predicts LONG / FLAT / SHORT
  setups per ticker (see the *AI Analysis Pipeline* section below)
- **Risk Management**: Position sizing, stop losses, take profit targets
- **IBKR Multi-Port Support**: Tries multiple configured IB ports for robust connection
- **Bracket Orders**: Automatic entry, stop loss, and take profit orders
- **Market Hours Detection**: Only trades during NYSE market hours
- **Email Alerts**: Automated notifications for trades, errors, and daily summaries
- **Git Integration**: Auto-updates from repository during off-hours
- **Environment Configuration**: Dev/Prod configs based on Git branch
- **Comprehensive Logging**: Detailed logs with timestamps and standardized logger usage across all modules
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
│   ├── retest_200ma/
│   │   ├── indicators.py      # Technical indicators
│   │   ├── trend_detector.py  # Pattern detection logic
│   │   └── validators.py      # Trend validation
│   └── ai_analysis/           # AI feature pipeline + trainers + orchestrator
│       ├── data_preparation/
│       │   ├── price_features.py      # returns, MA distance, daily range
│       │   ├── volume_features.py     # volume ratios, OBV slope
│       │   ├── indicator_features.py  # RSI, MACD, Bollinger, ATR, 200MA slope
│       │   └── feature_builder.py     # combines extractors + thermometer binarisation + sliding windows
│       ├── rbm_trainer.py             # wraps ai_modules/rbm for stock data
│       ├── cnn_trainer.py             # wraps ai_modules/cnn for supervised classification
│       └── ai_analyzer.py             # top-level orchestrator (build_dataset / train / predict)
├── ai_modules/                        # Low-level AI model implementations
│   ├── rbm/
│   │   ├── my_RBM_tf2_test.py         # Restricted Boltzmann Machine (TensorFlow 2)
│   │   ├── results/                   # RBM training artifacts (auto-created when AI is trained)
│   │   │   ├── logs/                  # TensorBoard event files
│   │   │   └── models/                # Per-epoch RBM weights (.h5)
│   │   └── ...                        # BAS dataset helpers, sampling utilities
│   └── cnn/
│       └── convolution_neural_network.py  # 1-D CNN (PyTorch) with RBM feature concat
├── results/                   # RBM training artifacts (auto-created when AI is trained)
│   ├── logs/                  # TensorBoard event files
│   └── models/                # Per-epoch RBM weights (.h5)
├── utils/
│   ├── alerts.py         # Alert system
│   ├── git_manager.py    # Git operations
│   ├── logger.py         # Logging utilities
│   └── metrics.py        # Performance metrics
├── tests/                # Regression tests
├── data/                 # Runtime data storage (auto-created)
│   ├── bot_logs/         # Log files for runs
│   ├── test_logs/        # Log files for test runs
│   └── performance/      # Performance data
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

## 🤖 AI Analysis Pipeline

An optional, decoupled machine learning pipeline that learns recurring patterns
in stock price / volume / indicator data and produces a per-ticker LONG / FLAT /
SHORT classification. It is designed as a **companion signal** to the deterministic
retest_200ma strategy and does **not** place orders on its own.

### Architecture

```
OHLCV bars (IB)
    │
    ▼
┌──────────────────────────────────────────┐
│ data_preparation/                        │
│   PriceFeatureExtractor    (8 features)  │
│   VolumeFeatureExtractor   (4 features)  │
│   IndicatorFeatureExtractor(5 features)  │
└──────────────────────────────────────────┘
    │                      (continuous)
    ▼
FeatureBuilder
    ├─ fit_bin_edges()   quantile bin edges learned across all tickers
    ├─ binarize()        thermometer encoding (n_bits per feature)
    └─ build_windows()   sliding windows of length window_size
    │
    ├──────► rbm_x  (N, window_size × 17 × n_bits)  uint8
    ├──────► cnn_x  (N, window_size × 17)           float32
    └──────► labels {0: SHORT, 1: FLAT, 2: LONG}    from fwd return
    │
    ▼
┌──────────────────────────────────────────┐
│ RBMTrainer   (unsupervised)              │
│   visible units = binarised feature bits │
│   → hidden activations used as features  │
└──────────────────────────────────────────┘
    │                       (hidden_features)
    ▼
┌──────────────────────────────────────────┐
│ CNNTrainer   (supervised, 3-class)       │
│   input: cnn_x (1-D signal) + RBM feats  │
│   output: softmax over {SHORT,FLAT,LONG} │
└──────────────────────────────────────────┘
    │
    ▼
AIAnalyzer.predict(symbol) → {'class', 'probs'}
```

### Feature Set (17 scale-free features per bar)

| Group | Features |
|-------|----------|
| **Price**      | `log_return_1d`, `log_return_5d`, `close_vs_ma20`, `close_vs_ma50`, `close_vs_ma200`, `daily_range_pct`, `close_to_high_pct`, `close_to_low_pct` |
| **Volume**     | `volume_ratio_20`, `volume_ratio_50`, `volume_log_change`, `obv_slope_20` |
| **Indicators** | `rsi14`, `macd_hist_norm`, `bb_position`, `atr14_pct`, `ma200_slope_pct` |

All features are computed in a scale-free form so different tickers can be
pooled into a single training corpus.

### Labels (for the CNN)

Each window is labelled by the forward return between the last bar of the
window and `forward_horizon` bars later:

| Forward return | Class |
|---|---|
| `> label_threshold`          | 2 = LONG  |
| `[-label_threshold, +thr]`   | 1 = FLAT  |
| `< -label_threshold`         | 0 = SHORT |

Defaults: `forward_horizon = 5` days, `label_threshold = 0.01` (1 %).

### Example: Training and Predicting

```python
from ib_insync import IB
from data_fetch.historical_data import StockDataFetcher
from data_fetch.stock_fetcher import StockTickerFetcher
from strategy.ai_analysis import AIAnalyzer, FeatureBuilder

# 1. Connect to IB (use a different clientId than the live bot)
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=9)

# 2. Build the fetcher and ticker list
fetcher = StockTickerFetcher()
data = StockDataFetcher(ib, config, params)
tickers = fetcher.stock_list[:100]   # smaller subset while iterating

# 3. Configure the pipeline
analyzer = AIAnalyzer(
    stock_data=data,
    feature_builder=FeatureBuilder(window_size=10, n_bits=4, forward_horizon=5),
    rbm_hidden_dim=64,
    rbm_epochs=30,
    cnn_epochs=20,
)

# 4. Train
analyzer.train(tickers, val_split=0.2)

# 5. Predict on a new ticker
print(analyzer.predict('AAPL'))
# -> {'symbol': 'AAPL', 'class': 'LONG', 'class_id': 2,
#     'probs': {'SHORT': 0.12, 'FLAT': 0.31, 'LONG': 0.57}}
```

### Configuration knobs

| Parameter | Where | Effect |
|---|---|---|
| `window_size`      | `FeatureBuilder` | Consecutive days per training sample. Drives RBM `visible_dim` and CNN `input_length`. |
| `n_bits`           | `FeatureBuilder` | Thermometer-encoding resolution per continuous feature. Higher = more detail but larger `visible_dim`. |
| `forward_horizon`  | `FeatureBuilder` | Bars ahead used to label each window. |
| `label_threshold`  | `FeatureBuilder` | Forward-return cutoff that separates LONG / SHORT from FLAT. |
| `rbm_hidden_dim`   | `AIAnalyzer`     | Size of the learned representation passed to the CNN. |
| `rbm_epochs`       | `AIAnalyzer`     | Contrastive-divergence epochs. |
| `cnn_epochs`       | `AIAnalyzer`     | Supervised training epochs. |

### Training artifacts

Running `analyzer.train(...)` writes to a `results/` directory relative to the
current working directory:

```
results/
├── logs/<date>/<time>/train/   # TensorBoard events from the RBM
├── models/<date>/…model.h5     # RBM weights snapshot per epoch
├── past_machines.csv           # index of saved RBMs
└── temp_<T>_<shape>_<mode>.csv # per-epoch weights / biases / temperature trace
```

### Operational notes

- **More data is better**: with the bot's default `lookback_days = 250`, the
  200-day MA features eat ~220 bars of warm-up, leaving roughly 15 usable
  windows per ticker. For serious training, increase `lookback_days` (e.g. to
  2000) or pull history from a separate long-history source.
- **Validation split is per-ticker, not per-day**: samples are concatenated in
  ticker order and the last `val_split` fraction becomes the held-out set — so
  the val set consists of *entire tickers* the models never saw. This is good
  for generalisation checks, less so for detecting regime drift.
- **Class imbalance**: the FLAT class usually dominates with default settings.
  Watch the `class counts` line logged by `AIAnalyzer.build_dataset()` and tune
  `label_threshold` / `forward_horizon` if the imbalance is severe.
- **Use a dedicated `clientId`** when running AI training so it does not
  interfere with a live bot session.

## 🛠️ Installation

### Prerequisites
- Python 3.10+
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

The `requirements.txt` includes the AI pipeline dependencies (TensorFlow,
PyTorch, scikit-learn, deepdish/h5py, tqdm, seaborn, cmocean). If you do not
plan to use the AI pipeline you can comment those lines out in the file — the
core trading bot itself does not import them.

### 3. Configure Environment (Optional)
For email alerts, create a `.env` file in the project root:
```
GMAIL_USER=your-gmail@gmail.com
GMAIL_PASSWORD=your-app-password
```

### 3. Set Up Interactive Brokers

1. **Download TWS or IB Gateway**
   - Visit interactivebrokers.com
   - Download Trader Workstation (TWS) or IB Gateway

2. **Configure API Access**
   - Launch TWS/IB Gateway
   - File → Global Configuration → API → Settings
   - ✅ Enable ActiveX and Socket Clients
   - Set the socket port to the value(s) configured in `config/dev.json` or `config/prod.json`
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
        "ma_slope_period": 20,
        "min_uptrend_slope": -0.01,
        "max_downtrend_slope": 0.01,
        "risk_reward_ratio": 2.0,
        "lookback_days": 250,
        "min_breakout_volume": 1.7,
        "min_breakout_strength": 0.7,
        "min_bounce_strength": 0.02,
        "max_retest_volume_ratio": 0.5,
        "max_retest_volume_absolute": 0.8,
        "max_days_since_retest": 3,
        "retest_distance": 0.005
    },
    "risk_management": {
        "risk_per_trade_pct": 0.05,
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
    "ports": [4002, 7497],
    "client_id": 1
  }
}
```
The bot will attempt each configured port in order until it connects successfully.
Use the same format in your environment files:
- `config/dev.json`: typically `"ports": [4002, 7497]`
- `config/prod.json`: typically `"ports": [4001, 7496]`

### Email Alerts Configuration (Optional)
To enable email notifications, add to your config files:
```json
{
  "alerts": {
    "enabled": true,
    "email": "your-email@example.com"
  }
}
```

Set environment variables for Gmail:

Unix/macOS:
```bash
export GMAIL_USER="your-gmail@gmail.com"
export GMAIL_PASSWORD="your-app-password"
```

Windows PowerShell:
```powershell
$env:GMAIL_USER = "your-gmail@gmail.com"
$env:GMAIL_PASSWORD = "your-app-password"
```

Or create a `.env` file in the project root:
```
GMAIL_USER=your-gmail@gmail.com
GMAIL_PASSWORD=your-app-password
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
- ✅ Sends email alerts for trades and errors (if configured)
- ✅ Monitors existing positions
- ✅ Auto-commits changes to Git
- ✅ Logs all activity

### Monitor Activity
The bot creates timestamped log files in `data/bot_logs/`:
```
data/bot_logs/trading_bot_3-22-2026_14-30.log
```

Example log output:
```
2026-03-22 14:30:00 - INFO - Starting trading bot...
2026-03-22 14:30:01 - INFO - Connected to IB at 127.0.0.1:4002
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
python tests_main.py
```

### Test Coverage
- `test_ai_analysis.py`: Tests data flow of the AI modules
- `test_position_manager.py`: Position tracking logic
- `test_retest_200ma.py`: Strategy pattern detection
- `test_risk_manager.py`: Risk calculation validation

### Monitor Activity
The tests creates timestamped log files in `data/test_logs/`:
```
data/test_logs/test_retest_200ma_3-22-2026_14-30.log
```

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
- Verify correct port (7497 for paper, 7496 for live)

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

**"FileNotFoundError" for stock_list.txt or positions.json**
- Ensure `data/` directory exists (auto-created on first run)
- Check file permissions
- Verify correct working directory when running the bot

**"Gmail authentication failed"**
- Use Gmail App Passwords (not regular password)
- Enable 2FA on Gmail account
- Check environment variables: `GMAIL_USER` and `GMAIL_PASSWORD`

### AI Pipeline Issues
**`ModuleNotFoundError: No module named 'tensorflow'` / `torch` / `deepdish`**
- Install AI dependencies: `pip install -r requirements.txt`
- These are only needed if you use `strategy.ai_analysis.AIAnalyzer`; the
  deterministic bot does not import them

**"No tickers produced usable features"**
- Each ticker needs at least ~220 bars of history before the 200-day MA
  features come online. Raise `lookback_days` in `trading_params.json` or
  use a longer history source

**RuntimeError: `x_train must be (N, <visible_dim>)`**
- The RBM's `visible_dim` must equal `window_size × n_features × n_bits`.
  If you supply a custom set of extractors, pass them all to the
  `FeatureBuilder(extractors=[...])` so its `visible_dim` property matches

**CNN validation accuracy stuck around 1/3**
- Class imbalance — most samples are FLAT. Check the `class counts=[…]`
  line logged during `build_dataset()` and tune `label_threshold` /
  `forward_horizon`

**`results/` directory filling up with .h5 files**
- The legacy RBM calls `save_model()` every epoch by design. Clear it between
  runs or run training inside a dedicated working directory

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
tail -f data/bot_logs/trading_bot_*.log

# Count signals found
grep "Signal found" data/bot_logs/trading_bot_*.log | wc -l

# Check position entries
grep "Entered position" data/bot_logs/trading_bot_*.log
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
2. Verify IB Gateway or TWS is running and API enabled
3. Ensure markets are open (for live data)
4. Review IB API documentation: https://interactivebrokers.github.io/

## Disclaimer

This is for educational purposes. Trading involves risk. Test thoroughly with paper trading before using real money. Past performance doesn't guarantee future results.