# Security Policy

## Important Notice

This project interacts with live brokerage accounts and real financial markets. Security vulnerabilities could result in unauthorized trades or financial loss.

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Instead, email **mohamedbouchtout@gmail.com** with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if you have one)

You should receive a response within 48 hours. Please allow time to investigate and patch before public disclosure.

## Scope

The following are in scope for security reports:

- Authentication or credential exposure (API keys, `.env` leaks)
- Unauthorized order placement or position manipulation
- Injection vulnerabilities in configuration parsing
- Insecure data handling (positions, account info)
- Docker or deployment misconfigurations that expose the bot

## Best Practices for Users

- **Never commit your `.env` file** (it's in `.gitignore` by default)
- **Use paper trading** to test any changes before going live
- **Run TWS/Gateway with "Allow connections from localhost only"** enabled
- **Use a dedicated IB account** for the bot, not your primary trading account
- **Review all code changes** before pulling updates to a live instance
