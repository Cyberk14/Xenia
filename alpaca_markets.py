import alpaca_trade_api as tradeapi
import logging
import json
import csv
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import pandas as pd

class AlpacaTrader:
    """
    A trading module for Alpaca Markets that handles BUY and SELL signals.
    - BUY: Purchases specified quantity of shares
    - SELL: Sells all shares of the specified symbol (using offshore storage quantity)
    - Stores positions offshore in JSON and CSV files
    """
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets", 
                 storage_dir: str = "offshore_positions"):
        """
        Initialize the Alpaca trader.
        
        Args:
            api_key (str): Alpaca API key
            secret_key (str): Alpaca secret key
            base_url (str): Base URL for Alpaca API (default: paper trading)
            storage_dir (str): Directory to store offshore position files
        """
        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )
        
        # Set up storage directory
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Storage file paths
        self.json_file = os.path.join(storage_dir, "positions.json")
        self.csv_file = os.path.join(storage_dir, "positions.csv")
        self.trades_csv = os.path.join(storage_dir, "trades_history.csv")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage files
        self._initialize_storage()
        
        # Verify account connection
        try:
            account = self.api.get_account()
            self.logger.info(f"Connected to Alpaca account: {account.id}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    def _initialize_storage(self):
        """Initialize storage files if they don't exist."""
        # Initialize JSON file
        if not os.path.exists(self.json_file):
            initial_data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "positions": {},
                "metadata": {
                    "created": datetime.now(timezone.utc).isoformat(),
                    "total_trades": 0
                }
            }
            with open(self.json_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
        
        # Initialize CSV files
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'symbol', 'quantity', 'avg_price', 'current_price', 'market_value',
                    'unrealized_pl', 'unrealized_pl_pct', 'side', 'last_updated'
                ])
        
        if not os.path.exists(self.trades_csv):
            with open(self.trades_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'action', 'quantity', 'price', 'order_id',
                    'status', 'total_cost', 'notes'
                ])
    
    def _load_positions_json(self) -> Dict[str, Any]:
        """Load positions from JSON file."""
        try:
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading JSON positions: {e}")
            return {"positions": {}, "metadata": {"total_trades": 0}}
    
    def _save_positions_json(self, data: Dict[str, Any]):
        """Save positions to JSON file."""
        try:
            data["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(self.json_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving JSON positions: {e}")
    
    def _update_position_storage(self, symbol: str, position_data: Optional[Dict[str, Any]] = None):
        """Update position in both JSON and CSV storage."""
        try:
            # Get current position from Alpaca if not provided
            if position_data is None:
                position_data = self.get_position(symbol)
            
            # Load current JSON data
            json_data = self._load_positions_json()
            
            # Update or remove position
            if position_data and position_data['qty'] != 0:
                # Update position
                json_data['positions'][symbol] = {
                    'symbol': symbol,
                    'quantity': position_data['qty'],
                    'avg_price': position_data['avg_entry_price'],
                    'market_value': position_data['market_value'],
                    'unrealized_pl': position_data['unrealized_pl'],
                    'unrealized_pl_pct': (position_data['unrealized_pl'] / abs(position_data['market_value'] - position_data['unrealized_pl'])) * 100 if position_data['market_value'] != position_data['unrealized_pl'] else 0,
                    'side': position_data['side'],
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }
            else:
                # Remove position if quantity is 0 or None
                json_data['positions'].pop(symbol, None)
            
            # Save JSON
            self._save_positions_json(json_data)
            
            # Update CSV
            self._update_csv_positions()
            
        except Exception as e:
            self.logger.error(f"Error updating position storage for {symbol}: {e}")
    
    def _update_csv_positions(self):
        """Update the CSV file with current positions."""
        try:
            json_data = self._load_positions_json()
            
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'symbol', 'quantity', 'avg_price', 'current_price', 'market_value',
                    'unrealized_pl', 'unrealized_pl_pct', 'side', 'last_updated'
                ])
                
                for symbol, pos in json_data['positions'].items():
                    # Calculate current price
                    current_price = (pos['market_value'] / pos['quantity']) if pos['quantity'] != 0 else 0
                    
                    writer.writerow([
                        symbol,
                        pos['quantity'],
                        pos['avg_price'],
                        current_price,
                        pos['market_value'],
                        pos['unrealized_pl'],
                        pos['unrealized_pl_pct'],
                        pos['side'],
                        pos['last_updated']
                    ])
                    
        except Exception as e:
            self.logger.error(f"Error updating CSV positions: {e}")
    
    def _log_trade(self, symbol: str, action: str, quantity: int, price: float = 0, 
                   order_id: str = "", status: str = "", notes: str = ""):
        """Log trade to trades history CSV."""
        try:
            total_cost = quantity * price if price > 0 else 0
            
            with open(self.trades_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    symbol,
                    action,
                    quantity,
                    price,
                    order_id,
                    status,
                    total_cost,
                    notes
                ])
                
            # Update trade counter in JSON
            json_data = self._load_positions_json()
            json_data['metadata']['total_trades'] = json_data['metadata'].get('total_trades', 0) + 1
            self._save_positions_json(json_data)
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
    
    def get_stored_positions(self) -> Dict[str, Any]:
        """Get all stored positions from offshore storage."""
        return self._load_positions_json()
    
    def get_stored_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stored position for a specific symbol."""
        json_data = self._load_positions_json()
        return json_data['positions'].get(symbol.upper())
    
    def sync_positions_from_alpaca(self):
        """Sync all positions from Alpaca to offshore storage."""
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                position_data = {
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'side': position.side,
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'unrealized_pl': float(position.unrealized_pl)
                }
                self._update_position_storage(position.symbol, position_data)
            
            self.logger.info("Successfully synced all positions from Alpaca")
            
        except Exception as e:
            self.logger.error(f"Error syncing positions from Alpaca: {e}")
    
    def export_positions_to_excel(self, filename: str = None):
        """Export positions to Excel file."""
        try:
            if filename is None:
                filename = os.path.join(self.storage_dir, f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            
            # Read positions CSV
            df_positions = pd.read_csv(self.csv_file)
            
            # Read trades history CSV
            df_trades = pd.read_csv(self.trades_csv)
            
            # Create Excel writer
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_positions.to_excel(writer, sheet_name='Positions', index=False)
                df_trades.to_excel(writer, sheet_name='Trades_History', index=False)
            
            self.logger.info(f"Positions exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {e}")
            return None
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            account = self.api.get_account()
            return {
                'account_id': account.id,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'equity': float(account.equity),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for a symbol from Alpaca.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict with position info or None if no position
        """
        try:
            position = self.api.get_position(symbol)
            return {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'side': position.side,
                'market_value': float(position.market_value),
                'avg_entry_price': float(position.avg_entry_price),
                'unrealized_pl': float(position.unrealized_pl)
            }
        except Exception as e:
            self.logger.info(f"No position found for {symbol}: {e}")
            return None
    
    def buy_signal(self, symbol: str, quantity: int) -> Dict[str, Any]:
        """
        Execute a BUY signal for specified quantity.
        
        Args:
            symbol (str): Stock symbol to buy
            quantity (int): Number of shares to buy
            
        Returns:
            Dict with order information
        """
        try:
            # Validate inputs
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            # Submit market buy order
            order = self.api.submit_order(
                symbol=symbol.upper(),
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            result = {
                'status': 'success',
                'action': 'BUY',
                'symbol': symbol.upper(),
                'quantity': quantity,
                'order_id': order.id,
                'submitted_at': order.submitted_at.isoformat(),
                'message': f"Buy order submitted for {quantity} shares of {symbol.upper()}"
            }
            
            # Log trade to offshore storage
            self._log_trade(
                symbol=symbol.upper(),
                action='BUY',
                quantity=quantity,
                order_id=order.id,
                status='SUBMITTED',
                notes=f"Market buy order for {quantity} shares"
            )
            
            # Update position storage (will be updated again when order fills)
            self._update_position_storage(symbol.upper())
            
            self.logger.info(result['message'])
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'action': 'BUY',
                'symbol': symbol.upper(),
                'quantity': quantity,
                'error': str(e),
                'message': f"Failed to submit buy order for {symbol.upper()}: {e}"
            }
            
            self.logger.error(error_result['message'])
            return error_result
    
    def sell_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Execute a SELL signal - sells ALL shares of the specified symbol.
        Gets quantity from offshore storage first, falls back to Alpaca if needed.
        
        Args:
            symbol (str): Stock symbol to sell
            
        Returns:
            Dict with order information
        """
        try:
            # First try to get quantity from offshore storage
            stored_position = self.get_stored_position(symbol)
            quantity_to_sell = 0
            
            if stored_position and stored_position['quantity'] > 0:
                quantity_to_sell = int(stored_position['quantity'])
                self.logger.info(f"Using stored quantity {quantity_to_sell} for {symbol.upper()}")
            else:
                # Fallback to Alpaca position if not in offshore storage
                alpaca_position = self.get_position(symbol)
                if alpaca_position and alpaca_position['qty'] > 0:
                    quantity_to_sell = int(alpaca_position['qty'])
                    self.logger.info(f"Using Alpaca quantity {quantity_to_sell} for {symbol.upper()}")
                else:
                    result = {
                        'status': 'no_action',
                        'action': 'SELL',
                        'symbol': symbol.upper(),
                        'quantity': 0,
                        'message': f"No position found for {symbol.upper()} to sell (checked both offshore storage and Alpaca)"
                    }
                    self.logger.info(result['message'])
                    return result
            
            # Check if we have shares to sell
            if quantity_to_sell <= 0:
                result = {
                    'status': 'no_action',
                    'action': 'SELL',
                    'symbol': symbol.upper(),
                    'quantity': quantity_to_sell,
                    'message': f"No shares to sell for {symbol.upper()}"
                }
                self.logger.info(result['message'])
                return result
            
            # Submit market sell order for all shares
            order = self.api.submit_order(
                symbol=symbol.upper(),
                qty=quantity_to_sell,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            result = {
                'status': 'success',
                'action': 'SELL',
                'symbol': symbol.upper(),
                'quantity': quantity_to_sell,
                'order_id': order.id,
                'submitted_at': order.submitted_at.isoformat(),
                'message': f"Sell order submitted for all {quantity_to_sell} shares of {symbol.upper()}"
            }
            
            # Log trade to offshore storage
            self._log_trade(
                symbol=symbol.upper(),
                action='SELL',
                quantity=quantity_to_sell,
                order_id=order.id,
                status='SUBMITTED',
                notes=f"Market sell order for all {quantity_to_sell} shares (from offshore storage)"
            )
            
            # Update position storage (position will be removed when order fills)
            self._update_position_storage(symbol.upper())
            
            self.logger.info(result['message'])
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'action': 'SELL',
                'symbol': symbol.upper(),
                'error': str(e),
                'message': f"Failed to submit sell order for {symbol.upper()}: {e}"
            }
            
            self.logger.error(error_result['message'])
            return error_result
    
    def execute_signal(self, signal: str, symbol: str, quantity: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a trading signal.
        
        Args:
            signal (str): 'BUY' or 'SELL'
            symbol (str): Stock symbol
            quantity (int, optional): Quantity for BUY signal (ignored for SELL)
            
        Returns:
            Dict with execution result
        """
        signal = signal.upper()
        
        if signal == 'BUY':
            if quantity is None:
                return {
                    'status': 'error',
                    'message': 'Quantity required for BUY signal'
                }
            return self.buy_signal(symbol, quantity)
        
        elif signal == 'SELL':
            return self.sell_signal(symbol)
        
        else:
            return {
                'status': 'error',
                'message': f'Invalid signal: {signal}. Use BUY or SELL'
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            Dict with order status
        """
        try:
            order = self.api.get_order(order_id)
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': order.qty,
                'status': order.status,
                'filled_qty': order.filled_qty,
                'filled_avg_price': order.filled_avg_price,
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                'filled_at': order.filled_at.isoformat() if order.filled_at else None
            }
        except Exception as e:
            return {
                'error': str(e),
                'message': f'Failed to get order status: {e}'
            }


# Example usage
def main():
    """Example usage of the AlpacaTrader module."""
    
    # Initialize trader (replace with your actual API credentials)
    trader = AlpacaTrader(
        api_key="PKB6827AE6J1CM0IKLEJ",
        secret_key="bRQuhjlrbz7uVqX3SeBfJ2KaRWsOcAYLXv5rbgZV",
        base_url="https://paper-api.alpaca.markets",  # Paper trading
        storage_dir="my_offshore_positions"  # Custom storage directory
    )
    
    # Sync existing positions from Alpaca
    trader.sync_positions_from_alpaca()
    
    # Get account info
    account = trader.get_account_info()
    print(f"Account info: {account}")
    
    # View stored positions
    stored_positions = trader.get_stored_positions()
    print(f"Stored positions: {stored_positions}")
    
    # Example BUY signal
    buy_result = trader.execute_signal('BUY', 'AAPL', 10)
    print(f"Buy result: {buy_result}")
    
    # Example SELL signal (sells all shares using offshore storage quantity)
    sell_result = trader.execute_signal('SELL', 'AAPL')
    print(f"Sell result: {sell_result}")
    
    # Export positions to Excel
    excel_file = trader.export_positions_to_excel()
    print(f"Positions exported to: {excel_file}")
    
    # Check order status if order was successful
    if buy_result.get('status') == 'success':
        order_status = trader.get_order_status(buy_result['order_id'])
        print(f"Order status: {order_status}")
    
    # View specific stored position
    aapl_position = trader.get_stored_position('AAPL')
    print(f"AAPL stored position: {aapl_position}")


if __name__ == "__main__":
    main()