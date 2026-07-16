# Troubleshooting

## Connection Issues

### "Connection Refused"

- Ensure TWS or IB Gateway is running
- Check that API access is enabled in TWS/Gateway settings
- Verify at least one port in `config.json` matches your TWS/Gateway (`4002`/`7497` for paper, `4001`/`7496` for live)
- Check that "Allow connections from localhost" is enabled

### Connection Drops During Off-Hours

IB drops idle connections. The bot automatically reconnects when this happens. If you see frequent reconnect cycles, this is normal behavior when the market is closed.

### "No Market Data"

- Accept market data agreements in IBKR Account Management
- Markets must be open for real-time data (9:30 AM - 4:00 PM ET weekdays)
- The bot uses delayed/free data by default (`reqMarketDataType(3)`)

## Trading Issues

### "Position Size Too Small"

- Account balance is too low relative to the risk parameters
- Increase account size or reduce `risk_per_trade_pct` in `trading_params.json`

### "No Signals Found"

- The 200 MA breakout/retest pattern is relatively rare — it may take days to find one
- Check logs to confirm stocks are being scanned (`Testing <TICKER>...`)
- Verify historical data is loading (no repeated `Failed to get data` warnings)

### Bracket Order Not Filling

- The parent order uses a market order during RTH only (`outsideRth=False`)
- If submitted near market close, it may not fill before the session ends
- The bot waits up to 10 minutes for a fill, then cancels the entire bracket

## Configuration Issues

### "Failed to load configuration"

- Check JSON syntax in `config/config.json` (missing commas, trailing commas)
- Ensure `config/config.json` exists

### "FileNotFoundError" for stock_list.txt or positions.json

- The `data/` directory is auto-created on first run
- Check file permissions
- Ensure you're running from the project root directory

### "Gmail authentication failed"

- Use a Gmail **App Password**, not your regular password
- Enable 2FA on your Gmail account first
- Check that `GMAIL_USER` and `GMAIL_PASSWORD` environment variables are set (or in `.env`)

## AI Pipeline Issues

### ModuleNotFoundError for torch

- Install dependencies: `pip install -r requirements.txt`
- PyTorch is required for the AI analysis pipeline (LSTM and CNN models)

### "No tickers produced usable features"

- Each ticker needs at least ~220 bars of history before the 200-day MA features are valid
- Increase `lookback_days` in `trading_params.json` (the AI analyzer defaults to 2000)

### "Failed to fetch market data"

- The `MarketFeatureExtractor` fetches VIX and SPY data via yfinance on startup
- If it fails (network issue), it falls back to neutral defaults — predictions still work but without market regime context
- This is a warning, not an error

### Validation accuracy stuck around 33%

- Class imbalance — most samples are FLAT
- Volatility-adjusted labels (default) usually produce more balanced classes than fixed thresholds
- Try adjusting `volatility_threshold` in `FeatureBuilder` (lower = more LONG/SHORT labels)
- Check the `class counts=[...]` log line during `build_dataset()`

### Early stopping triggers immediately

- Validation set may be too small — increase `lookback_days` to get more training samples
- Try increasing `patience` (default 5) in the trainer

### Switching between LSTM and CNN

- Set `model_type='cnn'` or `model_type='lstm'` when creating `AIAnalyzer`
- Default is `'lstm'` — LSTM generally performs better on sequential financial data
- Both models use the same features and labels, so results are directly comparable
