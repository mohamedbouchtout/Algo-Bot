#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
CONFIG_FILE="config/config.json"
ENV_FILE=".env"

echo "==========================================="
echo "  Algo-Bot Setup"
echo "==========================================="
echo ""

# --- Check Python ---
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.11 or higher is required but not found."
    echo "Please install Python from https://www.python.org/downloads/"
    exit 1
fi

echo "Found Python: $($PYTHON --version)"

# --- Create virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# --- Activate venv ---
source "$VENV_DIR/bin/activate"

# --- Install dependencies ---
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo "Dependencies installed."

# --- Configure email alerts ---
if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "==========================================="
    echo "  Email Alert Configuration"
    echo "==========================================="
    echo ""
    echo "The bot can send email alerts for trades, errors, and daily summaries."
    echo "This requires a Gmail account with an App Password."
    echo "(See: https://support.google.com/accounts/answer/185833)"
    echo ""

    read -rp "Do you want to configure email alerts? (y/n): " configure_email

    if [[ "$configure_email" =~ ^[Yy]$ ]]; then
        echo ""
        read -rp "Gmail address (sender): " gmail_user
        read -rsp "Gmail App Password: " gmail_password
        echo ""
        read -rp "Recipient email (where alerts are sent): " recipient_email

        # Write .env file
        cat > "$ENV_FILE" << EOF
GMAIL_USER=$gmail_user
GMAIL_PASSWORD=$gmail_password
EOF
        echo ""
        echo ".env file created."

        # Update config.json with recipient email
        RECIPIENT_EMAIL="$recipient_email" "$PYTHON" -c "
import json, os
with open('$CONFIG_FILE') as f:
    config = json.load(f)
config['alerts']['email'] = os.environ['RECIPIENT_EMAIL']
config['alerts']['enabled'] = True
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=4)
"
        echo "config.json updated with alert settings."
    else
        echo ""
        echo "Skipping email configuration."
        echo "You can configure it later by creating a .env file (see .env.example)."
    fi
else
    echo ""
    echo ".env file already exists, skipping email configuration."
fi

echo ""
echo "==========================================="
echo "  Setup Complete!"
echo "==========================================="
echo ""
echo "To start the bot:"
echo "  ./run.sh"
echo ""
echo "Make sure IB Gateway or TWS is running with API enabled before starting."
echo ""
