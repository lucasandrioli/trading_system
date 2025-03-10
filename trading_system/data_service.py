import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from trading_system.core.data.data_loader import DataLoader
from trading_system.utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class DataService:
    """Service for handling data operations and caching."""
    
    def __init__(self, data_dir: str, cache_manager: CacheManager):
        self.data_dir = data_dir
        self.cache_manager = cache_manager
        
        # Configure API keys if available
        from trading_system.core.analysis.qualitative_analysis import APIClient
        APIClient.POLYGON_API_KEY = self._get_polygon_api_key()
        
        # Configure DataLoader
        DataLoader.POLYGON_API_KEY = self._get_polygon_api_key()
    
    def _get_polygon_api_key(self) -> str:
        """Get Polygon API key from environment or configuration."""
        import os
        return os.environ.get('POLYGON_API_KEY', '')
    
    def get_historical_data(self, ticker: str, days: int = 60, interval: str = "1d", 
                           extended_hours: bool = False) -> pd.DataFrame:
        """Get historical data for a ticker with caching."""
        cache_key = f"hist_data:{ticker}:{days}:{interval}:{extended_hours}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            try:
                # Convert cached data back to DataFrame
                return pd.DataFrame(cached_data)
            except Exception as e:
                logger.warning(f"Error loading cached data for {ticker}: {e}")
        
        # Get fresh data
        df = DataLoader.get_asset_data(ticker, days, interval, extended_hours=extended_hours)
        
        if not df.empty:
            # Store in cache
            try:
                self.cache_manager.set(cache_key, df.to_dict())
            except Exception as e:
                logger.warning(f"Error caching data for {ticker}: {e}")
        
        return df
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices for multiple tickers with caching."""
        if not tickers:
            return {}
        
        # Check cache for each ticker
        cached_prices = {}
        tickers_to_fetch = []
        
        for ticker in tickers:
            cache_key = f"current_price:{ticker}"
            cached_price = self.cache_manager.get(cache_key)
            
            if cached_price is not None:
                cached_prices[ticker] = cached_price
            else:
                tickers_to_fetch.append(ticker)
        
        # Fetch only what's needed
        if tickers_to_fetch:
            fresh_prices = DataLoader.get_realtime_prices_bulk(tickers_to_fetch)
            
            # Cache new prices
            for ticker, price in fresh_prices.items():
                cache_key = f"current_price:{ticker}"
                self.cache_manager.set(cache_key, price, 300)  # 5 minute cache
            
            # Combine with cached prices
            return {**cached_prices, **fresh_prices}
        
        return cached_prices
    
    def update_prices(self, portfolio: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Update current prices for a portfolio."""
        if not portfolio or 'positions' not in portfolio:
            return portfolio, {}
        
        # Get all tickers
        tickers = list(portfolio['positions'].keys())
        
        # Get current prices
        prices = self.get_current_prices(tickers)
        
        # Update portfolio with current prices
        for ticker, price in prices.items():
            if ticker in portfolio['positions']:
                portfolio['positions'][ticker]['current_price'] = price
                # Update current position value
                qty = portfolio['positions'][ticker].get('quantity', 0)
                portfolio['positions'][ticker]['current_position'] = qty * price
        
        return portfolio, prices
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate a ticker symbol with caching."""
        cache_key = f"ticker_valid:{ticker}"
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # Check ticker validity
        is_valid = DataLoader.check_ticker_valid(ticker)
        
        # Cache result
        self.cache_manager.set(cache_key, is_valid, 86400)  # 1 day cache
        
        return is_valid
    
    def get_ticker_data_as_dict(self, ticker: str, days: int = 60, interval: str = "1d") -> Dict[str, Any]:
        """Get ticker data as dictionary format for API/JSON responses."""
        df = self.get_historical_data(ticker, days, interval)
        
        if df.empty:
            return {"success": False, "message": "No data available for this ticker"}
        
        try:
            # Convert to serializable format
            df_dict = df.reset_index()
            if 'Date' in df_dict.columns:
                df_dict['Date'] = df_dict['Date'].dt.strftime('%Y-%m-%d')
            
            data = df_dict.to_dict(orient='records')
            return {"success": True, "data": data}
        except Exception as e:
            logger.error(f"Error converting ticker data to dict for {ticker}: {e}")
            return {"success": False, "message": str(e)}
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get data for key market references like indices."""
        cache_key = "market_reference_data"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            from trading_system.core.analysis.market_analysis import MarketAnalysis
            data = MarketAnalysis.get_market_data()
            
            # Cache for 1 hour
            self.cache_manager.set(cache_key, data, 3600)
            
            return data
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    def clear_ticker_cache(self, ticker: str) -> None:
        """Clear all cached data for a specific ticker."""
        # Clear various caches for this ticker
        prefixes = [
            f"hist_data:{ticker}",
            f"current_price:{ticker}",
            f"ticker_valid:{ticker}",
            f"indicator_analysis:{ticker}",
            f"fundamental_analysis:{ticker}",
            f"news_analysis:{ticker}"
        ]
        
        for prefix in prefixes:
            for key in [k for k in self.cache_manager.keys() if k.startswith(prefix)]:
                self.cache_manager.delete(key)
        
        logger.info(f"Cleared cache for ticker {ticker}")