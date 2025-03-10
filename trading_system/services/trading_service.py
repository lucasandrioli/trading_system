import logging
import os
import json
from typing import Dict, Any

logger = logging.getLogger("trading_system.services.trading_service")

class TradingService:
    """Service for trading operations and parameters."""
    
    def __init__(self, config, cache_service):
        self.config = config
        self.cache_service = cache_service
        self.data_dir = config.get('DATA_FOLDER', 'data')
        self.params_file = os.path.join(self.data_dir, 'trading_params.json')
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_trading_parameters(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.params_file):
                with open(self.params_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading trading parameters: {e}")
        
        # Return default parameters
        return {
            "sma_period": 20,
            "ema_period": 9,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_window": 20,
            "bb_std": 2,
            "decision_buy_threshold": 60,
            "decision_sell_threshold": -60,
            "take_profit_pct": 5.0,
            "stop_loss_pct": -8.0,
            "trailing_stop_pct": 3.0,
            "weight_tech": 0.70,
            "weight_qual": 0.20,
            "weight_fund": 0.05,
            "weight_market": 0.05
        }
    
    def update_trading_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current_params = self.get_trading_parameters()
            updated_params = {**current_params, **params}
            
            with open(self.params_file, 'w') as f:
                json.dump(updated_params, f, indent=2)
            
            return {"success": True, "message": "Trading parameters updated successfully"}
        except Exception as e:
            logger.error(f"Error updating trading parameters: {e}")
            return {"success": False, "message": str(e)}