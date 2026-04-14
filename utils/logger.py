"""
Logger setup with rotation to manage log file size and retention.
"""

import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os

def setup_logger(config):
    """Setup logger with rotation"""
    
    # Get log directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, 'data', 'bot_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file path
    log_file = os.path.join(log_dir, 'trading_bot.log')
    
    # Get log level from config
    log_level = config.get('logging', {}).get('level', 'INFO')
    console_enabled = config.get('logging', {}).get('console', False)
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove any existing handlers (in case function is called multiple times)
    logger.handlers.clear()
    
    # FILE HANDLER - Rotates daily at midnight, keeps 30 days
    file_handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',       # Rotate at midnight
        interval=1,            # Every 1 day
        backupCount=30,        # Keep 30 days of logs
        encoding='utf-8'
    )
    
    # Format: 2026-04-06 14:30:45 - INFO - Message here
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    logger.addHandler(file_handler)
    
    # CONSOLE HANDLER (optional, for development)
    if console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
    
    logger.info("=" * 60)
    logger.info("Logger initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Rotation: Daily at midnight, keeping 30 days")
    logger.info("=" * 60)
    
    return logger