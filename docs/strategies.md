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

A machine learning pipeline that classifies each ticker as LONG, FLAT, or SHORT based on learned patterns in price, volume, technical indicators, and market regime features. The default model is an LSTM (Long Short-Term Memory) network, with a CNN option available.

### Architecture

```text
OHLCV bars (via yfinance)
    |
    v
Feature Extractors (20 scale-free features per bar)
    |- PriceFeatureExtractor      (8 features)
    |- VolumeFeatureExtractor     (4 features)
    |- IndicatorFeatureExtractor  (5 features)
    |- MarketFeatureExtractor     (3 features: VIX, SPY trend, SPY vs 200MA)
    |
    v
FeatureBuilder
    |- build_windows()   sliding windows of 10 consecutive days
    |- Volatility-adjusted labels (forward return / ATR)
    |
    v
LSTMTrainer or CNNTrainer (PyTorch, selected via model_type)
    |- LSTM: processes window as a sequence (10 timesteps x 20 features)
    |- CNN:  processes window as a flattened 1-D signal (200 values)
    |- Early stopping with best-weight restore
    |- Weight decay regularization
    |
    v
AIAnalyzer.predict(symbol) --> {'class': 'LONG', 'probs': {...}}
```

### Model Types

The `AIAnalyzer` supports two model architectures via the `model_type` parameter:

| Model | `model_type` | How it works | Best for |
|-------|-------------|--------------|----------|
| **LSTM** (default) | `'lstm'` | Processes the 10-day window as a sequence, learning temporal dependencies between consecutive days | Capturing sequential patterns (e.g., 3-day momentum followed by a reversal) |
| **CNN** | `'cnn'` | Processes the window as a flattened 1-D signal through convolutional filters | Detecting local patterns regardless of position in the window |

The LSTM is the default because financial time series are inherently sequential — the order of bars matters, and LSTMs are designed to model that.

### Feature Set (20 features per bar)

| Group | Features |
|-------|----------|
| **Price** (8) | `log_return_1d`, `log_return_5d`, `close_vs_ma20`, `close_vs_ma50`, `close_vs_ma200`, `daily_range_pct`, `close_to_high_pct`, `close_to_low_pct` |
| **Volume** (4) | `volume_ratio_20`, `volume_ratio_50`, `volume_log_change`, `obv_slope_20` |
| **Indicators** (5) | `rsi14`, `macd_hist_norm`, `bb_position`, `atr14_pct`, `ma200_slope_pct` |
| **Market Regime** (3) | `vix_normalized`, `spy_50d_return`, `spy_vs_ma200` |

All features are scale-free so different tickers can be pooled into a single training corpus. Market regime features provide context about the broader market (bull/bear, high/low volatility).

### Volatility-Adjusted Labels

Labels are generated from forward returns normalized by each stock's volatility:

```text
adjusted_return = forward_return / ATR(14)%

If adjusted_return > volatility_threshold  → LONG  (2)
If adjusted_return < -volatility_threshold → SHORT (0)
Otherwise                                 → FLAT  (1)
```

This means a 1% move on a low-volatility utility stock is treated as a stronger signal than a 1% move on a high-volatility biotech stock. The default `volatility_threshold` is `1.0` (one ATR).

Fixed-threshold labels (`label_threshold = 0.01`) are available by setting `volatility_adjusted_labels=False` in `FeatureBuilder`.

### Training

The bot trains AI models automatically on weekends when the market is closed. Training is done per-sector — each sector gets its own model trained on all tickers in that sector.

The training pipeline:
1. Fetches historical data for every ticker in the sector (via yfinance)
2. Extracts 20 continuous features per bar and pools them
3. Builds 10-day sliding windows with volatility-adjusted labels
4. Trains the LSTM (or CNN) with early stopping and weight decay
5. Best model weights are restored after training

### Walk-Forward Cross-Validation

For model evaluation, use `walk_forward_train()` instead of the standard `train()`:

```python
analyzer.walk_forward_train(n_splits=5)
```

This runs expanding-window validation: train on the first 50% of data, validate on the next 10%, then expand the training window and repeat. Each fold trains a fresh model, and per-fold accuracy is reported. After all folds, a final model is trained on all data for live prediction.

### Early Stopping

Both the LSTM and CNN trainers use early stopping:
- Tracks validation loss each epoch
- If no improvement for `patience` epochs (default 5), training stops
- Restores the best model weights (lowest validation loss)
- Prevents overfitting on small datasets

### Standalone Usage

```python
from data_fetch.historical_data import StockDataFetcher
from strategy.ai_analysis.ai_analyzer import AIAnalyzer

data = StockDataFetcher()
analyzer = AIAnalyzer(
    stock_data=data,
    model_type='lstm',      # or 'cnn'
    cnn_epochs=20,
    params=params,
)

analyzer.train(['AAPL', 'MSFT', 'GOOGL', ...], val_split=0.2)
print(analyzer.predict('AAPL'))
# {'symbol': 'AAPL', 'class': 'LONG', 'class_id': 2,
#  'probs': {'SHORT': 0.12, 'FLAT': 0.31, 'LONG': 0.57}}

# Or with walk-forward validation:
metrics = analyzer.walk_forward_train(n_splits=5)
print(f"Avg accuracy: {metrics['avg_accuracy']:.3f}")
```

### Configuration Knobs

| Parameter | Location | Effect |
|---|---|---|
| `model_type` | `AIAnalyzer` | `'lstm'` (default) or `'cnn'` |
| `window_size` | `FeatureBuilder` | Consecutive days per training sample (default 10) |
| `forward_horizon` | `FeatureBuilder` | Bars ahead used to label each window (default 5) |
| `volatility_threshold` | `FeatureBuilder` | ATR-normalized return threshold for labels (default 1.0) |
| `volatility_adjusted_labels` | `FeatureBuilder` | `True` (default) to normalize returns by volatility |
| `cnn_epochs` | `AIAnalyzer` | Max training epochs (early stopping may end sooner) |
| `patience` | `CNNTrainer`/`LSTMTrainer` | Epochs without improvement before stopping (default 5) |
| `confidence_threshold` | `trading_params.json` | Minimum softmax probability to generate a signal |

### Backtesting

The backtest (`tests/legacy/test_ai_backtest.py`) simulates live trading on historical data using the same per-sector model architecture:

1. Fetches historical data and splits into train/test periods (75/25)
2. Trains one AI model per sector on the training period
3. Walks forward through the test period day by day
4. Makes predictions using the correct sector's model
5. Simulates bracket-order trades (entry, stop loss, take profit)
6. Reports per-trade P&L, win rate, total return, profit factor, max drawdown, and per-sector breakdown

### Notes

- **More data is better**: Use `lookback_days = 1825` or more for AI training.
- **Class imbalance**: Volatility-adjusted labels produce more balanced classes than fixed thresholds. Watch the `class counts` log line.
- Historical data is fetched via yfinance (no IB connection or rate limits required for data).
