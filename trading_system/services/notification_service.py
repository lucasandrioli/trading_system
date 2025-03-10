import logging
import os
import json
from typing import Dict, Any

logger = logging.getLogger("trading_system.services.notification_service")

class NotificationService:
    """Service for sending notifications about portfolio events."""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('DATA_FOLDER', 'data')
        self.config_file = os.path.join(self.data_dir, 'notification_config.json')
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_notification_config(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading notification config: {e}")
        
        # Default configuration
        return {
            "enabled": False,
            "methods": {
                "email": {"enabled": False, "address": ""},
                "webhook": {"enabled": False, "url": ""}
            },
            "notify_trades": True,
            "notify_thresholds": True
        }
    
    def notify_trade(self, ticker: str, action: str, quantity: int, price: float, reason: str) -> bool:
        logger.info(f"Trade notification: {action} {quantity} {ticker} @ ${price:.2f} - {reason}")
        return True
    
    def notify_trade_execution(self, trade_info: Dict[str, Any]) -> bool:
        logger.info(f"Trade execution: {trade_info}")
        return True