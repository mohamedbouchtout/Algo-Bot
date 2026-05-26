# Troubleshooting

## Connection Issues

### "Connection Refused"

- Ensure TWS or IB Gateway is running
- Check that API access is enabled in TWS/Gateway settings
- Verify the port matches your config (`4002`/`7497` for paper, `4001`/`7496` for live)
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

- Check JSON syntax in config files (missing commas, trailing commas)
- Ensure `config/dev.json` and `config/prod.json` exist
- Verify git is installed (used for branch detection to select config)

### "FileNotFoundError" for stock_list.txt or positions.json

- The `data/` directory is auto-created on first run
- Check file permissions
- Ensure you're running from the project root directory

### "Gmail authentication failed"

- Use a Gmail **App Password**, not your regular password
- Enable 2FA on your Gmail account first
- Check that `GMAIL_USER` and `GMAIL_PASSWORD` environment variables are set (or in `.env`)

## AI Pipeline Issues

### ModuleNotFoundError for tensorflow / torch / deepdish

- Install AI dependencies: `pip install -r requirements.txt`
- These are only needed if you use the AI analysis pipeline; the core bot doesn't import them

### "No tickers produced usable features"

- Each ticker needs at least ~220 bars of history before the 200-day MA features are valid
- Increase `lookback_days` in `trading_params.json` (the AI analyzer defaults to 2000)

### RuntimeError: "x_train must be (N, visible_dim)"

- The RBM's `visible_dim` must equal `window_size x n_features x n_bits`
- If using custom extractors, pass them all to `FeatureBuilder(extractors=[...])`

### CNN validation accuracy stuck around 33%

- Class imbalance — most samples are FLAT
- Check the `class counts=[...]` log line during `build_dataset()`
- Tune `label_threshold` (try 0.005 or 0.02) and `forward_horizon`

### results/ directory filling up with .h5 files

- The RBM saves a model snapshot every epoch by design
- Clear the directory between training runs, or run training in a dedicated working directory
