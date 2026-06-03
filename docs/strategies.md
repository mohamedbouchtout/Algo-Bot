# Strategies

The bot supports two strategies that can work independently or together. When both produce a signal for the same ticker, the AI signal takes priority.

## 200 MA Breakout & Retest

The primary strategy detects breakout and retest patterns around the 200-day moving average.

### Long Setup

1. **Breakout**: Price crosses above the 200 MA with high volume (1.7x+ average)
2. **Retest**: Price pulls back to within 0.5% of the 200 MA on low volume
3. **Bounce**: Price moves upward off the MA with positive momentum
4. **Entry**: Long position at current price, stop below the retest low, target at 2:1 R:R

### Short Setup

1. **Breakdown**: Price crosses below the 200 MA with high volume
2. **Retest**: Price rallies back to within 0.5% of the 200 MA on low volume
3. **Rejection**: Price moves downward off the MA with negative momentum
4. **Entry**: Short position at current price, stop above the retest high, target at 2:1 R:R

### Validation Checks

Before generating a signal, the detector validates:

- **MA trend direction**: MA must be trending in the signal direction (or flat)
- **Breakout volume**: Must exceed `min_breakout_volume` times average volume
- **Breakout candle strength**: Body must fill at least 70% of the candle range
- **Retest volume**: Must be below the breakout volume (weak counter-pressure)
- **Bounce strength**: Minimum price movement off the MA after retest
- **Recency**: Retest must have occurred within `max_days_since_retest` days
- **Risk cap**: Per-share risk must not exceed 5% of entry price

### Dynamic Stop Loss

Stop losses are calculated using the daily range and VIX:

- **VIX > 25** (high volatility): Wider stop = daily range x (ATR multiplier + 0.5)
- **VIX < 15** (low volatility): Tighter stop = daily range x (ATR multiplier - 0.5)
- **Normal**: Stop = daily range x ATR multiplier

---

## AI Analysis Pipeline

An optional machine learning pipeline that classifies each ticker as LONG, FLAT, or SHORT based on learned patterns in price, volume, and technical indicators.

### Architecture

```
OHLCV bars (from yfinance)
    |
    v
Feature Extractors (17 scale-free features per bar)
    |
    v
FeatureBuilder
    |- fit_bin_edges()     quantile-based binning across all tickers
    |- binarize()          thermometer encoding (n_bits per feature)
    |- build_windows()     sliding windows of length window_size
    |
    |--- rbm_x   (N, window_size x 17 x n_bits)  uint8     --> RBM input
    |--- cnn_x   (N, window_size x 17)            float32   --> CNN input
    |--- labels  {0: SHORT, 1: FLAT, 2: LONG}               from forward return
    |
    v
RBMTrainer (unsupervised, TensorFlow)
    |- Learns compressed hidden features from binarised windows
    |
    v
CNNTrainer (supervised, PyTorch)
    |- Input: continuous features + RBM hidden features
    |- Output: softmax over {SHORT, FLAT, LONG}
    |
    v
AIAnalyzer.predict(symbol) --> {'class': 'LONG', 'probs': {...}}
```

### Feature Set (17 features per bar)

| Group | Features |
|-------|----------|
| **Price** | `log_return_1d`, `log_return_5d`, `close_vs_ma20`, `close_vs_ma50`, `close_vs_ma200`, `daily_range_pct`, `close_to_high_pct`, `close_to_low_pct` |
| **Volume** | `volume_ratio_20`, `volume_ratio_50`, `volume_log_change`, `obv_slope_20` |
| **Indicators** | `rsi14`, `macd_hist_norm`, `bb_position`, `atr14_pct`, `ma200_slope_pct` |

All features are scale-free so different tickers can be pooled into a single training corpus.

### Labels

Each sliding window is labelled by the forward return over `forward_horizon` bars:

| Forward Return | Class |
|---|---|
| > `label_threshold` | 2 = LONG |
| Between -threshold and +threshold | 1 = FLAT |
| < -`label_threshold` | 0 = SHORT |

Defaults: `forward_horizon = 5` days, `label_threshold = 0.01` (1%).

### Training

The bot trains AI models automatically on weekends when the market is closed. Training is done per-sector — each sector gets its own RBM + CNN trained on all tickers in that sector.

The training pipeline:
1. Fetches historical data for every ticker in the sector
2. Extracts continuous features and pools them
3. Fits quantile-based bin edges across the pooled data
4. Trains the RBM (unsupervised feature learning)
5. Extracts hidden features from the RBM
6. Trains the CNN on continuous features + RBM features with forward-return labels

### Standalone Usage

```python
from data_fetch.historical_data import StockDataFetcher
from strategy.ai_analysis.ai_analyzer import AIAnalyzer
from strategy.ai_analysis.data_preparation.feature_builder import FeatureBuilder

data = StockDataFetcher()
analyzer = AIAnalyzer(
    stock_data=data,
    feature_builder=FeatureBuilder(window_size=10, n_bits=4),
    rbm_hidden_dim=64,
    rbm_epochs=30,
    cnn_epochs=20,
    params=params,
)

analyzer.train(['AAPL', 'MSFT', 'GOOGL', ...], val_split=0.2)
print(analyzer.predict('AAPL'))
# {'symbol': 'AAPL', 'class': 'LONG', 'class_id': 2,
#  'probs': {'SHORT': 0.12, 'FLAT': 0.31, 'LONG': 0.57}}
```

### Configuration Knobs

| Parameter | Location | Effect |
|---|---|---|
| `window_size` | `FeatureBuilder` | Consecutive days per training sample |
| `n_bits` | `FeatureBuilder` | Thermometer-encoding resolution per feature |
| `forward_horizon` | `FeatureBuilder` | Bars ahead used to label each window |
| `label_threshold` | `FeatureBuilder` | Forward-return cutoff separating LONG/SHORT from FLAT |
| `rbm_hidden_dim` | `AIAnalyzer` | Size of learned representation passed to CNN |
| `rbm_epochs` | `AIAnalyzer` | Contrastive-divergence training epochs |
| `cnn_epochs` | `AIAnalyzer` | Supervised training epochs |
| `confidence_threshold` | `trading_params.json` | Minimum softmax probability to generate a signal |

### Training Artifacts

Training writes to a `results/` directory:

```
results/
├── logs/           # TensorBoard event files
└── models/         # Per-epoch RBM weight snapshots (.h5)
```

### Notes

- **More data is better**: The default `lookback_days = 250` leaves only ~15 usable windows per ticker after MA warm-up. For better training, use `lookback_days = 2000` or more.
- **Class imbalance**: FLAT usually dominates. Watch the `class counts` log line and tune `label_threshold` / `forward_horizon` if needed.
- Historical data is fetched via yfinance (no IB connection or rate limits required for data).
