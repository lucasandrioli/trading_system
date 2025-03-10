import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.analysis.market_analysis import MarketAnalysis
from data.data_loader import DataLoader
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class MarketService:
    """Service for market-related operations like sentiment analysis and reference data."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self._market_sentiment_timestamp = None
    
    def get_market_sentiment(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get market sentiment with caching."""
        cache_key = "market_sentiment"
        now = datetime.now().timestamp()
        
        if not force_refresh:
            cached_sentiment = self.cache_manager.get(cache_key)
            cached_timestamp = self._market_sentiment_timestamp
            
            if cached_sentiment and cached_timestamp and (now - cached_timestamp < 1800):  # 30 minutes
                return cached_sentiment
        
        try:
            sentiment = MarketAnalysis.get_market_sentiment()
            self.cache_manager.set(cache_key, sentiment)
            self._market_sentiment_timestamp = now
            return sentiment
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            
            # Return cached if available
            cached_sentiment = self.cache_manager.get(cache_key)
            if cached_sentiment:
                return cached_sentiment
            
            # Return default
            return {"sentiment": "unknown", "trend": "unknown", "market_bias_score": 0}
    
    def get_ticker_data(self, ticker: str, days: int = 60, interval: str = "1d") -> Optional[Dict[str, Any]]:
        """Get historical data for a ticker."""
        cache_key = f"ticker_data:{ticker}:{days}:{interval}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            df = DataLoader.get_asset_data(ticker, days, interval)
            
            if df.empty:
                return None
            
            # Convert to serializable format
            df_dict = df.reset_index().to_dict(orient='records')
            
            # Cache the result
            self.cache_manager.set(cache_key, df_dict)
            
            return df_dict
        except Exception as e:
            logger.error(f"Error getting data for {ticker}: {e}")
            return None
    
    def get_market_reference_data(self) -> Dict[str, Any]:
        """Get data for key market references like indices and sectors."""
        cache_key = "market_reference_data"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            reference_data = {}
            reference_tickers = {
                "US_MARKET": "SPY",
                "TECH_SECTOR": "QQQ", 
                "BONDS": "TLT",
                "GOLD": "GLD",
                "VOLATILITY": "^VIX",
                "EMERGING": "EEM",
                "OIL": "USO",
                "CRYPTO": "BTC-USD"
            }
            
            for name, ticker in reference_tickers.items():
                df = DataLoader.get_asset_data(ticker, days=20)
                
                if df.empty:
                    continue
                
                last_close = float(df['Close'].iloc[-1])
                returns_1d = (last_close / float(df['Close'].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0
                returns_5d = (last_close / float(df['Close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
                
                reference_data[name] = {
                    "ticker": ticker,
                    "last_close": last_close,
                    "returns_1d": returns_1d,
                    "returns_5d": returns_5d
                }
            
            # Cache the result
            self.cache_manager.set(cache_key, reference_data, 3600)  # 1 hour expiry
            
            return reference_data
        except Exception as e:
            logger.error(f"Error getting market reference data: {e}")
            return {}
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if a ticker exists and has data available."""
        cache_key = f"ticker_valid:{ticker}"
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        try:
            is_valid = DataLoader.check_ticker_valid(ticker)
            self.cache_manager.set(cache_key, is_valid, 86400)  # Cache for 1 day
            return is_valid
        except Exception as e:
            logger.error(f"Error validating ticker {ticker}: {e}")
            return False
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices for multiple tickers."""
        if not tickers:
            return {}
        
        # Check cache first
        cached_prices = {}
        missing_tickers = []
        
        for ticker in tickers:
            cache_key = f"current_price:{ticker}"
            cached_price = self.cache_manager.get(cache_key)
            
            if cached_price is not None:
                cached_prices[ticker] = cached_price
            else:
                missing_tickers.append(ticker)
        
        # Get new prices in batch if needed
        if missing_tickers:
            try:
                new_prices = DataLoader.get_realtime_prices_bulk(missing_tickers)
                
                # Cache new prices
                for ticker, price in new_prices.items():
                    cache_key = f"current_price:{ticker}"
                    self.cache_manager.set(cache_key, price, 300)  # 5 minute expiry
                
                # Combine with cached prices
                return {**cached_prices, **new_prices}
            except Exception as e:
                logger.error(f"Error getting prices: {e}")
                return cached_prices
        
        return cached_prices