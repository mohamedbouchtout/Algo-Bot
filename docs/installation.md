# Installation

## Prerequisites

- Python 3.10+
- Interactive Brokers account (paper trading recommended for testing)
- TWS (Trader Workstation) or IB Gateway installed and running

## 1. Clone and Install

```bash
git clone https://github.com/mohamedbouchtout/Algo-Bot.git
cd Algo-Bot
pip install -r requirements.txt
```

`requirements.txt` includes AI pipeline dependencies (TensorFlow, PyTorch, scikit-learn). If you don't plan to use the AI pipeline, you can comment out those lines — the core bot doesn't import them.

## 2. Set Up Interactive Brokers

### Download TWS or IB Gateway

Visit [interactivebrokers.com](https://www.interactivebrokers.com/) and download Trader Workstation (TWS) or IB Gateway.

### Configure API Access

1. Launch TWS or IB Gateway
2. Go to **File > Global Configuration > API > Settings**
3. Enable **ActiveX and Socket Clients**
4. Set the socket port to match your config:
   - Paper trading: `4002` (Gateway) or `7497` (TWS)
   - Live trading: `4001` (Gateway) or `7496` (TWS)
5. Enable **Allow connections from localhost only**

### Paper Trading Account

1. Create a paper trading account at IBKR
2. Fund with virtual currency
3. Use paper trading credentials to log in

## 3. Email Alerts (Optional)

To receive email notifications for trades, errors, and daily summaries, create a `.env` file in the project root:

```
GMAIL_USER=your-gmail@gmail.com
GMAIL_PASSWORD=your-app-password
```

You must use a Gmail **App Password**, not your regular password. To generate one:

1. Enable 2FA on your Gmail account
2. Go to [Google App Passwords](https://myaccount.google.com/apppasswords)
3. Generate a new app password for "Mail"

Then enable alerts in your config file (`config/dev.json` or `config/prod.json`):

```json
{
  "alerts": {
    "enabled": true,
    "email": "your-email@example.com"
  }
}
```

## 4. Run the Bot

```bash
python main.py
```

The bot will:
1. Pull any pending git updates
2. Connect to IBKR (trying each configured port)
3. Load or train AI models if needed
4. Begin scanning during market hours

## Docker

See [Deployment](deployment.md) for Docker setup instructions.

## Verify It's Working

Check the log output in `data/bot_logs/`:

```
2026-03-22 14:30:00 - INFO - Starting trading bot...
2026-03-22 14:30:01 - INFO - Connected to IB at 127.0.0.1:4002
2026-03-22 14:30:02 - INFO - Scanning all stocks...
```
