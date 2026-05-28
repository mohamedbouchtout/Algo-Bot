# Deployment

## Local Machine

```bash
python main.py
```

The bot pauses automatically when the market is closed and resumes scanning when it opens. Your machine must stay running during market hours (9:30 AM - 4:00 PM ET).

## Cloud Server

On a VPS (AWS, DigitalOcean, etc.):

```bash
nohup python main.py > /dev/null 2>&1 &
```

All output goes to the rotating log files in `data/bot_logs/`, so redirecting stdout isn't necessary.

To monitor:

```bash
tail -f data/bot_logs/trading_bot.log
```

## Docker

### Build the Image

```bash
docker build -t algo-bot .
```

### Run with Auto-Restart

```bash
docker run --restart unless-stopped \
  -v "$PWD/config:/app/config" \
  -v "$PWD/data:/app/data" \
  algo-bot
```

The `unless-stopped` restart policy means:
- Container restarts on failure
- Container restarts on system reboot
- Container does NOT restart if you explicitly stop it

### Quick Test (No Restart)

```bash
docker run --rm algo-bot
```

### Docker Compose (Optional)

Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  algo-bot:
    build: .
    restart: unless-stopped
    volumes:
      - ./config:/app/config
      - ./data:/app/data
```

Then run:

```bash
docker-compose up -d
```

### Notes

- The `Dockerfile` uses `requirements.txt` to install dependencies
- `.dockerignore` is included to keep the image build clean
- Mount `config/` and `data/` as volumes to persist configuration and logs outside the container
- For production, always use a restart policy

## Monitoring

### Log Files

Logs rotate daily at midnight with 30-day retention:

```bash
# Follow live output
tail -f data/bot_logs/trading_bot.log

# Count signals found
grep "Signal found" data/bot_logs/trading_bot.log

# Check filled positions
grep "FILLED" data/bot_logs/trading_bot.log

# Check errors
grep "ERROR" data/bot_logs/trading_bot.log
```

### Positions

Active positions are tracked in `data/positions.json` and synced with IB on every connection.

### Email Alerts

When configured, the bot sends emails for:
- Bot started / stopped
- Trade entries and exits (with P&L)
- Critical errors
