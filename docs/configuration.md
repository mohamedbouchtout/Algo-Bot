# Configuration

The bot uses JSON files in the `config/` directory. The active config is selected automatically based on the current git branch.

## Environment Selection

| Branch | Config File | Typical Use |
|--------|------------|-------------|
| `bot/production` | `config/prod.json` | Live or paper trading in production |
| Any other branch | `config/dev.json` | Development and testing |

## IBKR Connection (`dev.json` / `prod.json`)

```json
{
  "ib": {
    "host": "127.0.0.1",
    "ports": [4002, 7497],
    "client_id": 1
  }
}
```

| Field | Description |
|-------|-------------|
| `host` | IB Gateway / TWS host address |
| `ports` | Ports to try in order. `4002`/`4001` = Gateway, `7497`/`7496` = TWS |
| `client_id` | API client ID. Use different IDs for simultaneous connections |

## Logging

```json
{
  "logging": {
    "level": "INFO",
    "console": true
  }
}
```

- `level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `console`: `true` to also print to stdout (useful for development)

Logs rotate daily at midnight with 30-day retention in `data/bot_logs/`.

## Email Alerts

```json
{
  "alerts": {
    "enabled": true,
    "email": "you@example.com"
  }
}
```

Requires `GMAIL_USER` and `GMAIL_PASSWORD` environment variables. See [Installation](installation.md) for setup.

## Git Integration

```json
{
  "git": {
    "enabled": true,
    "branch": "bot/nonproduction",
    "main_branch": "main",
    "commit_interval": 3600,
    "update_check_interval": 21600
  }
}
```

The bot checks for remote updates once per day after market close (4 PM ET). If new commits are found on `main_branch`, it pulls and restarts automatically.

## Trading Parameters (`trading_params.json`)

### 200 MA Strategy

```json
{
  "strategy_retest_200ma": {
    "ma_period": 200,
    "ma_slope_period": 20,
    "min_uptrend_slope": -0.01,
    "max_downtrend_slope": 0.01,
    "risk_reward_ratio": 2.0,
    "stop_loss_pct": 0.03,
    "lookback_days": 250,
    "min_breakout_volume": 1.7,
    "min_breakout_strength": 0.7,
    "min_bounce_strength": 0.02,
    "max_retest_volume_ratio": 0.5,
    "max_retest_volume_absolute": 0.8,
    "max_days_since_retest": 3,
    "retest_distance": 0.005,
    "ATR": 1.5
  }
}
```

| Parameter | Description |
|-----------|-------------|
| `ma_period` | Moving average period (days) |
| `ma_slope_period` | Window for calculating MA trend direction |
| `min_uptrend_slope` / `max_downtrend_slope` | Slope thresholds for trend validation |
| `risk_reward_ratio` | Minimum reward-to-risk ratio for entries |
| `lookback_days` | Historical data to fetch per stock |
| `min_breakout_volume` | Minimum volume ratio vs average for a valid breakout |
| `min_breakout_strength` | Minimum candle body ratio for breakout bar |
| `min_bounce_strength` | Minimum price bounce off the MA |
| `max_retest_volume_ratio` | Retest volume must be below this fraction of breakout volume |
| `max_days_since_retest` | Maximum days between retest and current bar |
| `retest_distance` | How close to the MA the price must get (as a fraction) |
| `ATR` | ATR multiplier for dynamic stop loss calculation |

### AI Analyzer

```json
{
  "ai_analyzer": {
    "confidence_threshold": 0.8,
    "risk_reward_ratio": 2.0,
    "stop_loss_pct": 0.03,
    "lookback_days": 2000,
    "ATR": 1.5
  }
}
```

| Parameter | Description |
|-----------|-------------|
| `confidence_threshold` | Minimum softmax probability to act on a prediction |
| `lookback_days` | Historical bars to fetch for AI training/prediction |

### Risk Management

```json
{
  "risk_management": {
    "risk_per_trade_pct": 0.03,
    "max_investment_pct": 0.70,
    "max_positions": 10,
    "max_position_pct": 0.20
  }
}
```

| Parameter | Description |
|-----------|-------------|
| `risk_per_trade_pct` | Max account equity risked per trade (3% = 0.03) |
| `max_investment_pct` | Max fraction of account invested at any time |
| `max_positions` | Max number of concurrent open positions |
| `max_position_pct` | Max fraction of account in a single position (caps penny stock sizing) |

### Timing

```json
{
  "timing": {
    "scan_interval": 1800,
    "market_check_interval": 900
  }
}
```

| Parameter | Description |
|-----------|-------------|
| `scan_interval` | Seconds between full stock scans during market hours |
| `market_check_interval` | Seconds between market-hours checks when market is closed |
