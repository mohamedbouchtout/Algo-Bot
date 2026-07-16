# Installation

## Prerequisites

- Python 3.11+
- Interactive Brokers account (paper trading recommended for testing)
- TWS (Trader Workstation) or IB Gateway installed and running

## 1. Clone and Setup

```bash
git clone https://github.com/mohamedbouchtout/Algo-Bot.git
cd Algo-Bot
./setup.sh        # Linux/Mac
setup.bat          # Windows
```

The setup script will:
1. Verify Python 3.11+ is installed
2. Create a virtual environment (`.venv`)
3. Install all dependencies from `requirements.txt`
4. Optionally configure email alerts (Gmail credentials + recipient email)

If you prefer to set up manually:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate.bat    # Windows
pip install -r requirements.txt
```

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

If you ran the setup script and chose to configure email alerts, this is already done. Otherwise, follow the steps below.

Create a `.env` file in the project root (see `.env.example` for a template):

```
GMAIL_USER=your-gmail@gmail.com
GMAIL_PASSWORD=your-app-password
```

You must use a Gmail **App Password**, not your regular password. To generate one:

1. Enable 2FA on your Gmail account
2. Go to [Google App Passwords](https://myaccount.google.com/apppasswords)
3. Generate a new app password for "Mail"

Then enable alerts in `config/config.json`:

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
./run.sh           # Linux/Mac
run.bat            # Windows
```

Or manually:

```bash
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate.bat       # Windows CMD
# .venv\Scripts\Activate.ps1       # Windows PowerShell
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
