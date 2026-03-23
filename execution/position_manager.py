"""
Track and manage positions, sync with IB
"""

import json
import os
import logging
from ib_insync import *
from typing import Dict

# Setup logging
logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, ib, config, params):
        self.ib = ib
        self.file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'positions.json')
        self.active_positions = self.load_positions()  # Load existing positions from JSON
        self.config = config
        self.params = params
    
    def monitor_positions(self):
        """Monitor and manage active positions"""
        # Get current positions from IB
        ib_positions = self.ib.positions()
        
        # Create set of symbols with actual positions (non-zero quantity)
        ib_symbols = {pos.contract.symbol for pos in ib_positions if pos.position != 0}
        
        # Remove closed positions from our tracking
        closed_positions = []
        for symbol in list(self.active_positions.keys()):
            if symbol not in ib_symbols:
                closed_positions.append(symbol)
                position_info = self.active_positions[symbol]
                
                # Log the closed position
                logging.info(
                    f"Position closed: {symbol} ({position_info['signal']['type']}) - "
                    f"Removing from active positions"
                )
                
                # Remove from tracking
                del self.active_positions[symbol]

                # Remove from JSON
                self.remove_position(symbol)
        
        # Log summary
        if closed_positions:
            logging.info(f"Removed {len(closed_positions)} closed positions: {closed_positions}")

        # Log currently active positions
        if self.active_positions:
            logging.info(f"Active positions: {len(self.active_positions)} stocks")
            for symbol, info in self.active_positions.items():
                # Find the position in IB data for P&L info
                ib_pos = next((p for p in ib_positions if p.contract.symbol == symbol), None)
                if ib_pos:
                    logging.info(
                        f"  {symbol}: {info['signal']['type']}, "
                        f"Qty: {ib_pos.position}, "
                        f"Avg Cost: ${ib_pos.avgCost:.2f}, "
                        f"Current: ${ib_pos.marketPrice:.2f}, "
                        f"P&L: ${ib_pos.unrealizedPNL:.2f}"
                    )
        else:
            logging.info("No active positions")

    def load_positions(self):
        """Load positions from JSON and sync with IB"""
        try:
            if not os.path.exists(self.file_path):
                logging.info("No positions.json file found - starting fresh")
                return
            
            with open(self.file_path, 'r') as file:
                data = json.load(file)
            
            if not data:
                logging.info("positions.json is empty - starting fresh")
                return
            
            # Get actual IB positions to verify
            ib_positions = self.ib.positions()
            ib_symbols = {pos.contract.symbol for pos in ib_positions if pos.position != 0}
            
            # Load positions from JSON
            loaded_count = 0
            symbols_to_remove = []
            for symbol, position_data in data.items():
                # Only load if position actually exists in IB
                if symbol in ib_symbols:
                    # Reconstruct signal dict
                    signal = {
                        'type': position_data['type'],
                        'symbol': symbol,
                        'entry': position_data['entry'],
                        'stop': position_data['stop'],
                        'target': position_data['target'],
                        'risk': position_data['risk'],
                        'reward': position_data['reward'],
                        'breakout_date': position_data['breakout_date'],
                        'retest_date': position_data['retest_date'],
                        'current_date': position_data['current_date'],
                        'breakout_volume_ratio': position_data.get('breakout_volume_ratio', 0),
                        'retest_volume_ratio': position_data.get('retest_volume_ratio', 0),
                        'avg_volume': position_data.get('avg_volume', 0),
                        'bounce_strength': position_data.get('bounce_strength', 0),
                        'breakdown_strength': position_data.get('breakdown_strength', 0),
                        'ma_slope': position_data.get('ma_slope', 0),
                        'ma_slope_pct': position_data.get('ma_slope_pct', 0)
                    }
                    
                    self.active_positions[symbol] = {
                        'signal': signal,
                        'shares': position_data['shares'],
                        'entry_time': position_data['entry_time']
                    }
                    
                    loaded_count += 1
                    logging.info(f"Loaded position from JSON: {symbol} ({position_data['type']})")
                else:
                    logging.warning(f"Position {symbol} in JSON but not in IB - removing from JSON")
                    symbols_to_remove.append(symbol)
            
            logging.info(f"Loaded {loaded_count} positions from positions.json")

            # Now safe to remove
            if symbols_to_remove:
                for symbol in symbols_to_remove:
                    del data[symbol]
                with open(self.file_path, 'w') as file:
                    json.dump(data, file, indent=4)
            
        except Exception as e:
            logging.error(f"Failed to load positions from JSON: {e}")
    
    def add_position(self, symbol: str, signal: Dict, shares: int, entry_time: str):
        """Add new position to tracking"""
        try:
            # Load existing data
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as file:
                    data = json.load(file)
            else:
                data = {}

            # Create entry with proper datetime serialization
            new_entry = {
                'type': signal['type'],
                'symbol': symbol,
                'entry': signal['entry'],
                'stop': signal['stop'],
                'target': signal['target'],
                'risk': signal['risk'],
                'reward': signal['risk'] * self.params['strategy_retest_200ma']['risk_reward_ratio'],
                'breakout_date': signal['breakout_date'].strftime('%Y-%m-%d') if hasattr(signal['breakout_date'], 'strftime') else str(signal['breakout_date']),
                'retest_date': signal['retest_date'].strftime('%Y-%m-%d') if hasattr(signal['retest_date'], 'strftime') else str(signal['retest_date']),
                'current_date': signal['current_date'].strftime('%Y-%m-%d') if hasattr(signal['current_date'], 'strftime') else str(signal['current_date']),
                'breakout_volume_ratio': signal.get('breakout_volume_ratio', 0),
                'retest_volume_ratio': signal.get('retest_volume_ratio', 0),
                'avg_volume': signal.get('avg_volume', 0),
                'bounce_strength': signal.get('bounce_strength', 0),
                'breakdown_strength': signal.get('breakdown_strength', 0),
                'ma_slope': signal.get('ma_slope', 0),
                'ma_slope_pct': signal.get('ma_slope_pct', 0),
                'shares': shares,
                'entry_time': entry_time
            }
            
            # Add to data
            data[symbol] = new_entry
            
            # Write back to file
            with open(self.file_path, 'w') as file:
                json.dump(data, file, indent=4)
            
            logging.debug(f"Saved position {symbol} to positions.json")
            
        except Exception as e:
            logging.error(f"Failed to save position to JSON: {e}")

    
    def remove_position(self, symbol: str):
        """Remove closed position"""
        try:
            if not os.path.exists(self.file_path):
                return
            
            with open(self.file_path, 'r') as file:
                data = json.load(file)
            
            if symbol in data:
                del data[symbol]
                
                with open(self.file_path, 'w') as file:
                    json.dump(data, file, indent=4)
                
                logging.debug(f"Removed position {symbol} from positions.json")
            else:
                logging.warning(f"{symbol} not found in positions.json for removal")
            
        except Exception as e:
            logging.error(f"Failed to remove position from JSON: {e}")
    
    def get_position_count(self) -> int:
        """Get number of active positions"""
        return len(self.active_positions)