"""
Track and manage positions, sync with IB
"""

import json
import logging
import os
from typing import Dict

from ib_insync import *

from utils.alerts import AlertManager

# Setup logging
logger = logging.getLogger()


class PositionManager:
    def __init__(self, ib, alert_manager: AlertManager, config, params):
        self.ib = ib
        self.file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'positions.json')
        self.active_positions = {}
        self.config = config
        self.params = params
        self.alert_manager = alert_manager

    def monitor_positions(self):
        """Monitor and manage active positions"""
        # Get current positions from IB
        ib_positions = self.ib.run(self.ib.reqPositionsAsync())
        ib_portfolio = self.ib.portfolio()

        # Create set of symbols with actual positions (non-zero quantity)
        ib_symbols = {pos.contract.symbol for pos in ib_positions if pos.position != 0}

        # Build a lookup for portfolio data (has PnL info)
        portfolio_by_symbol = {p.contract.symbol: p for p in ib_portfolio if p.position != 0}

        # Remove closed positions from our tracking
        closed_positions = []
        for symbol in list(self.active_positions.keys()):
            if symbol not in ib_symbols:
                closed_positions.append(symbol)
                position_info = self.active_positions[symbol]
                entry_price = position_info['signal']['entry']
                shares = position_info['shares']

                # Try to get realized PnL from portfolio snapshot taken before close
                ib_pos = portfolio_by_symbol.get(symbol)
                if ib_pos is not None:
                    pnl = float(ib_pos.realizedPNL) if ib_pos.realizedPNL else 0.0
                    exit_price = float(ib_pos.marketPrice)
                else:
                    pnl = 0.0
                    exit_price = None
                pnl_pct = (pnl / (entry_price * shares) * 100) if entry_price and shares else 0.0

                logger.info(f'Position closed: {symbol} ({position_info["signal"]["type"]}) - PnL: ${pnl:.2f} ({pnl_pct:.2f}%)')

                del self.active_positions[symbol]
                self.remove_position(symbol)
                self.alert_manager.alert_trade_exit(
                    symbol=symbol,
                    exit_type=position_info['signal']['type'],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    shares=shares,
                )

        # Log summary
        if closed_positions:
            logger.info(f'Removed {len(closed_positions)} closed positions: {closed_positions}')

        # Log currently active positions
        if self.active_positions:
            logger.info(f'Active positions: {len(self.active_positions)} stocks')

            for symbol, info in self.active_positions.items():
                ib_pos = portfolio_by_symbol.get(symbol)
                if ib_pos:
                    logger.info(
                        f'  {symbol}: {info["signal"]["type"]}, '
                        f'Qty: {ib_pos.position}, '
                        f'Avg Cost: ${ib_pos.averageCost:.2f}, '
                        f'Current: ${ib_pos.marketPrice:.2f}, '
                        f'P&L: ${ib_pos.unrealizedPNL:.2f}'
                    )
        else:
            logger.info('No active positions')

    def load_positions(self) -> dict:
        """Load positions from JSON and sync with IB"""
        out: dict = {}
        try:
            if not os.path.exists(self.file_path):
                logger.info('No positions.json file found - starting fresh')
                return out

            with open(self.file_path) as file:
                data = json.load(file)

            if not data:
                logger.info('positions.json is empty - starting fresh')
                return out

            # Get actual IB positions to verify
            self.ib.reqPositions()
            ib_positions = self.ib.run(self.ib.reqPositionsAsync())
            self.ib.sleep(3)
            ib_symbols = {pos.contract.symbol for pos in ib_positions if pos.position != 0}

            # Load positions from JSON
            loaded_count = 0
            symbols_to_remove = []
            for symbol, position_data in data.items():
                # Only load if position actually exists in IB
                if symbol in ib_symbols:
                    # Reconstruct signal dict
                    signal = None
                    if position_data['strategy_type'] == '200ma_retest':
                        signal = {
                            'strategy_type': position_data['strategy_type'],
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
                            'ma_slope_pct': position_data.get('ma_slope_pct', 0),
                        }
                    elif position_data['strategy_type'] == 'ai_analysis':
                        signal = {
                            'strategy_type': position_data['strategy_type'],
                            'type': position_data['type'],
                            'symbol': symbol,
                            'entry': position_data['entry'],
                            'stop': position_data['stop'],
                            'target': position_data['target'],
                            'risk': position_data['risk'],
                            'reward': position_data['reward'],
                            'confidence': position_data.get('confidence', 0),
                        }

                    out[symbol] = {'signal': signal, 'shares': position_data['shares'], 'entry_time': position_data['entry_time']}

                    loaded_count += 1
                    logger.info(f'Loaded position from JSON: {symbol} ({position_data["type"]})')
                else:
                    logger.warning(f'Position {symbol} in JSON but not in IB - removing from JSON')
                    symbols_to_remove.append(symbol)

            logger.info(f'Loaded {loaded_count} positions from positions.json')

            # Now safe to remove
            if symbols_to_remove:
                for symbol in symbols_to_remove:
                    del data[symbol]
                with open(self.file_path, 'w') as file:
                    json.dump(data, file, indent=4)

        except Exception as e:
            logger.error(f'Failed to load positions from JSON: {e}')
        return out

    def add_position(self, symbol: str, signal: dict, shares: int, entry_time: str):
        """Add new position to tracking"""
        try:
            # Load existing data
            if os.path.exists(self.file_path):
                with open(self.file_path) as file:
                    data = json.load(file)
            else:
                data = {}

            # Create entry with proper datetime serialization
            new_entry = None
            if signal['strategy_type'] == '200ma_retest':
                new_entry = {
                    'strategy_type': signal['strategy_type'],
                    'type': signal['type'],
                    'symbol': symbol,
                    'entry': signal['entry'],
                    'stop': signal['stop'],
                    'target': signal['target'],
                    'risk': signal['risk'],
                    'reward': signal['risk'] * self.params['strategy_retest_200ma']['risk_reward_ratio'],
                    'breakout_date': signal['breakout_date'].strftime('%Y-%m-%d')
                    if hasattr(signal['breakout_date'], 'strftime')
                    else str(signal['breakout_date']),
                    'retest_date': signal['retest_date'].strftime('%Y-%m-%d')
                    if hasattr(signal['retest_date'], 'strftime')
                    else str(signal['retest_date']),
                    'current_date': signal['current_date'].strftime('%Y-%m-%d')
                    if hasattr(signal['current_date'], 'strftime')
                    else str(signal['current_date']),
                    'breakout_volume_ratio': signal.get('breakout_volume_ratio', 0),
                    'retest_volume_ratio': signal.get('retest_volume_ratio', 0),
                    'avg_volume': signal.get('avg_volume', 0),
                    'bounce_strength': signal.get('bounce_strength', 0),
                    'breakdown_strength': signal.get('breakdown_strength', 0),
                    'ma_slope': signal.get('ma_slope', 0),
                    'ma_slope_pct': signal.get('ma_slope_pct', 0),
                    'shares': shares,
                    'entry_time': entry_time,
                }
            elif signal['strategy_type'] == 'ai_analysis':
                new_entry = {
                    'strategy_type': signal['strategy_type'],
                    'type': signal['type'],
                    'symbol': symbol,
                    'entry': signal['entry'],
                    'stop': signal['stop'],
                    'target': signal['target'],
                    'risk': signal['risk'],
                    'reward': signal['reward'],
                    'confidence': signal['confidence'],
                    'shares': shares,
                    'entry_time': entry_time,
                }

            # Add to data
            data[symbol] = new_entry

            # Write back to file
            with open(self.file_path, 'w') as file:
                json.dump(data, file, indent=4)

            logger.debug(f'Saved position {symbol} to positions.json')

        except Exception as e:
            logger.error(f'Failed to save position to JSON: {e}')

    def remove_position(self, symbol: str):
        """Remove closed position"""
        try:
            if not os.path.exists(self.file_path):
                return

            with open(self.file_path) as file:
                data = json.load(file)

            if symbol in data:
                del data[symbol]

                with open(self.file_path, 'w') as file:
                    json.dump(data, file, indent=4)

                logger.debug(f'Removed position {symbol} from positions.json')
            else:
                logger.warning(f'{symbol} not found in positions.json for removal')

        except Exception as e:
            logger.error(f'Failed to remove position from JSON: {e}')

    def get_position_count(self) -> int:
        """Get number of active positions"""
        return len(self.active_positions)
