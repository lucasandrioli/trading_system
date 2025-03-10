import logging
from typing import Dict, Any, Optional
from trading_system.core.data.data_loader import DataLoader

logger = logging.getLogger("trading_system.services.data_service")

class DataService:
    """Service for data operations."""
    
    def __init__(self, config, cache_service):
        self.config = config
        self.cache_service = cache_service
        api_key = config.get('POLYGON_API_KEY', '')
        if api_key:
            DataLoader.POLYGON_API_KEY = api_key
    
    def get_ticker_data(self, ticker: str, days: int = 60, interval: str = "1d") -> Dict[str, Any]:
        try:
            cache_key = f"ticker_data:{ticker}:{days}:{interval}"
            cached_data = self.cache_service.get(cache_key)
            
            if cached_data:
                return {"success": True, "data": cached_data}
            
            df = DataLoader.get_asset_data(ticker, days, interval)
            
            if df.empty:
                return {"success": False, "message": "No data available for this ticker"}
            
            # Convert to serializable format
            df_dict = df.reset_index()
            if 'Date' in df_dict.columns:
                df_dict['Date'] = df_dict['Date'].dt.strftime('%Y-%m-%d')
            
            data = df_dict.to_dict(orient='records')
            
            # Cache the result
            self.cache_service.set(cache_key, data)
            
            return {"success": True, "data": data}
        except Exception as e:
            logger.error(f"Error getting data for {ticker}: {e}")
            return {"success": False, "message": str(e)}
    
    def get_current_prices(self, tickers: list) -> Dict[str, float]:
        try:
            return DataLoader.get_realtime_prices_bulk(tickers)
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    def check_ticker_valid(self, ticker: str) -> bool:
        return DataLoader.check_ticker_valid(ticker)