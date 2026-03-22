"""
Trading bot entry point
"""

from core.bot import TradingBot

def main():
    # Create and run the bot
    bot = TradingBot()
    bot.run()

if __name__ == "__main__":
    main()