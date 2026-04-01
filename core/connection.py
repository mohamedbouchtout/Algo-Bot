"""
Connection manager for connecting to TWS.
"""

import logging
import os
import sys
from ib_insync import *
from execution.position_manager import PositionManager

# Setup logging
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self, ib, config, params):
        self.ib = ib
        self.config = config
        self.params = params
        self.host = config['ib']['host']
        self.ports = config['ib']['ports']
        self.port = 0  # Will be set during connection attempts
        self.client_id = config['ib']['client_id']

    def connect(self):
        """Connect to Interactive Brokers"""
        for port in self.ports:
            try:
                logging.info(f"Attempting connection to {self.host}:{port}...")
                
                # Try to connect
                self.ib.connect(self.host, port, clientId=self.client_id)
                self.port = port  # Store the successful port
                
                logging.info(f"Connected to IB at {self.host}:{port}")
                
                # Set market data type
                self.ib.reqMarketDataType(3)  # Delayed/free data
                
                # Load positions and STORE the manager
                position_manager = PositionManager(self.ib, self.config, self.params)
                position_manager.load_positions()
                
                return True
                
            except ConnectionRefusedError:
                logging.warning(f"Connection refused on port {port}, trying next...")
                continue
            except Exception as e:
                logging.error(f"Connection failed on port {port}: {e}")
                continue
        
        # If we get here, all ports failed
        logging.error(f"Failed to connect to any port: {self.ports}")
        logging.error("Make sure TWS or IB Gateway is running with API enabled")
        return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        self.ib.disconnect()
        logging.info("Disconnected from IB")

    def restart_bot(self):
        """Restart the bot to apply updates"""
        logging.info("Restarting bot to apply updates...")
        
        # Disconnect cleanly
        self.disconnect()
        
        # Restart the Python script
        python = sys.executable
        os.execv(python, [python] + sys.argv)
    
    def ensure_connected(self) -> bool:
        """Ensure IB connection is active, reconnect if needed"""
        if not self.ib.isConnected():
            logging.warning("IB connection lost. Reconnecting...")
            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self.ib.reqMarketDataType(3)
                logging.info("Reconnected to IB")
                return True
            except Exception as e:
                logging.error(f"Reconnection failed: {e}")
                return False
        return True