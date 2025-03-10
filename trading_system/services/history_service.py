import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger("trading_system.services.history_service")

class HistoryService:
    """Service for managing portfolio history."""
    
    def __init__(self, config, portfolio_service):
        self.config = config
        self.portfolio_service = portfolio_service
        self.data_dir = config.get('DATA_FOLDER', 'data')
        self.history_file = os.path.join(self.data_dir, 'portfolio_history.json')
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_portfolio_history(self) -> List[Dict[str, Any]]:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio history: {e}")
        return []
    
    def save_portfolio_history(self, history: List[Dict[str, Any]]) -> bool:
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving portfolio history: {e}")
            return False
    
    def add_portfolio_snapshot(self, portfolio=None, account_balance=None) -> Dict[str, Any]:
        try:
            # If portfolio not provided, load it
            if portfolio is None:
                loaded_portfolio = self.portfolio_service.load_portfolio()
                portfolio = loaded_portfolio.positions
                account_balance = loaded_portfolio.account_balance
            
            snapshot = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": 0,  # Simplified
                "cash_balance": account_balance,
                "total_value": account_balance,
            }
            
            # Add to history
            history = self.load_portfolio_history()
            history.append(snapshot)
            
            # Keep last 90 days only
            if len(history) > 90:
                history = history[-90:]
            
            self.save_portfolio_history(history)
            return snapshot
        except Exception as e:
            logger.error(f"Error adding portfolio snapshot: {e}")
            return {}