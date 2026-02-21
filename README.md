# Breakout & Retest Trading Bot

Automated trading bot that scans S&P 500 and NASDAQ stocks for 200-day moving average breakout and retest patterns.

## Strategy Overview

The bot looks for this pattern:
1. **Breakout**: Stock price crosses above/below the 200-day moving average
2. **Retest**: Price comes back to test the 200 MA
3. **Bounce**: Price bounces off the MA in the breakout direction
4. **Entry**: Enter position with 2:1 risk/reward ratio

### Long Setup
- Price breaks **above** 200 MA
- Price retests 200 MA from above
- Price bounces up
- Enter long with stop below retest low

### Short Setup
- Price breaks **below** 200 MA
- Price retests 200 MA from below  
- Price bounces down
- Enter short with stop above retest high

## Installation

### 1. Install Required Packages

```bash
pip install ib_insync pandas numpy requests beautifulsoup4
```

### 2. Set Up Interactive Brokers

1. Download and install TWS or IB Gateway
2. Open paper trading account at interactivebrokers.com
3. Enable API in TWS:
   - File → Global Configuration → API → Settings
   - ✅ Enable ActiveX and Socket Clients
   - Socket port: 7497 (paper trading)
   - ✅ Allow connections from localhost only
4. Restart TWS

### 3. Get Stock List (Optional)

To monitor the full S&P 500 and NASDAQ-100:

```bash
fetch_stock_list.py
```

This files `stocks.txt` with all tickers. The bot will automatically run and use this.

## Usage

### Start the Bot

Make sure TWS/IB Gateway is running and logged in, then:

```bash
python bot.py
```

The bot will:
- ✅ Run continuously 24/7
- ✅ Only trade during non-holiday market hours (9:30 AM - 4:00 PM EST)
- ✅ Scan all stocks every 5 minutes
- ✅ Automatically enter positions when signals are found
- ✅ Use bracket orders (entry + stop loss + take profit)
- ✅ Risk 1% of account per trade

### Monitor the Bot

The bot logs all activity to:
- **Console**: Real-time updates
- **trading_bot_<month>-<day>-<year>_<hour>-<minute>.log**: Full log file

Example log output:
```
2024-02-14 10:30:00 - INFO - Scanning 550 stocks...
2024-02-14 10:32:15 - INFO - Signal found: LONG AAPL @ $225.50
2024-02-14 10:32:16 - INFO - Entered LONG position in AAPL: 40 shares @ $225.50, Stop: $223.00, Target: $230.50
```

## Configuration

Edit `bot.py` to customize:

```python
# Trading parameters
self.ma_period = 200  # Moving average period
self.risk_reward_ratio = 2.0  # 2:1 reward to risk
self.scan_interval = 300  # Scan every 5 minutes

# Position sizing (in run method)
risk_per_trade = net_liq * 0.01  # Risk 1% per trade
```

## Risk Management

**Built-in safeguards:**
- ✅ 2:1 minimum risk/reward on every trade
- ✅ Automatic stop loss on all positions
- ✅ Automatic take profit targets
- ✅ Position sizing based on account risk (1% default)
- ✅ One position per stock maximum
- ✅ Paper trading mode by default

**To go live (only after thorough testing!):**
1. Change port from 7497 to 7496
2. Reduce risk per trade to 0.5% or lower
3. Start with small account
4. Monitor closely for first few weeks

## Important Notes

### Pattern Detection
The bot looks for breakouts in the last 20 days. It requires:
- Clear break above/below 200 MA
- Retest within 2% of 200 MA
- Confirmation bounce in breakout direction

### Limitations
- **Delayed data**: Uses 15-min delayed market data (free tier)
  - For real-time data, subscribe to IB market data
- **Scanning time**: Scanning 500+ stocks takes ~5 minutes
  - Considers historical patterns, not tick-by-tick
- **Execution**: Uses market orders for entry
  - May have slippage on less liquid stocks

### Running 24/7

**Local Computer:**
- Keep your PC running during market hours
- Bot automatically pauses when market is closed

**Cloud Server (Recommended):**
```bash
# On AWS/DigitalOcean VPS
nohup python bot.py > bot.log 2>&1 &
```

This keeps the bot running even if you disconnect.

## Troubleshooting

### "Connection Refused" Error
- TWS/IB Gateway isn't running
- API not enabled in settings
- Wrong port number

### "No Market Data" Error  
- Markets are closed (run during 9:30 AM - 4:00 PM EST weekdays)
- Or need to accept market data agreements in Account Management

### "Position Size Too Small" Warning
- Account balance too low for the risk amount
- Increase account size or adjust risk percentage

### Bot Not Finding Signals
- Pattern is relatively rare (might take days to find)
- Check logs to see which stocks were scanned
- Verify historical data is loading correctly

## Testing

Before running with real money:

1. **Paper trade for 30+ days** minimum
2. **Track performance**: Win rate, average R:R, drawdown
3. **Verify pattern quality**: Manually review signals
4. **Test edge cases**: What happens during volatile markets?
5. **Monitor logs**: Any errors or unexpected behavior?

## Files

- `bot.py` - Main bot script
- `_200ma_retest_detection.py` - Helper that does the 200 MA analysis
- `fetch_stock_list.py` - Helper to get S&P 500/NASDAQ tickers
- `stocks.txt` - Full list of stocks to monitor (generated)
- `trading_bot_<month>-<day>-<year>_<hour>-<minute>.log` - Activity log
- `README.md` - This file

## Example Workflow

**Monday 9:00 AM:**
1. Start TWS, log into paper trading
2. Run: `python bot.py`
3. Bot waits until 9:30 AM market open

**Monday 9:30 AM:**
- Bot begins scanning stocks
- Logs: "Scanning 550 stocks..."

**Monday 9:45 AM:**
- Bot finds signal: LONG AAPL
- Places bracket order automatically
- Logs position details

**Monday 4:00 PM:**
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