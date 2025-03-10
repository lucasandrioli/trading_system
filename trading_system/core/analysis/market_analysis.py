import logging
import numpy as np
from typing import Dict, Any
from trading_system.core.data.data_loader import DataLoader
from trading_system.core.analysis.technical_indicators import TechnicalIndicators

# Configure logging
logger = logging.getLogger("trading_system.market_analysis")

class MarketAnalysis:
    """Análise do sentimento do mercado."""
    
    MARKET_REFERENCE_TICKERS = {
        "US_MARKET": "SPY",
        "TECH_SECTOR": "QQQ",
        "BONDS": "TLT",
        "GOLD": "GLD",
        "VOLATILITY": "^VIX",
        "EMERGING": "EEM",
        "OIL": "USO",
        "CRYPTO": "BTC-USD"
    }
    
    @staticmethod
    def get_market_data() -> dict:
        """Obtém dados de mercado para referências-chave (índices, setores, etc.)."""
        market_data = {}
        
        for name, ticker in MarketAnalysis.MARKET_REFERENCE_TICKERS.items():
            try:
                df = DataLoader.get_asset_data(ticker, days=20, interval="1d", use_polygon=False)
                
                if df.empty or len(df) < 5:
                    logger.warning(f"Dados insuficientes para análise de mercado: {ticker}")
                    continue
                
                last_close = df['Close'].iloc[-1]
                returns_1d = (last_close / df['Close'].iloc[-2] - 1) * 100 if len(df) >= 2 else 0.0
                returns_5d = (last_close / df['Close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0.0
                
                rsi_series = TechnicalIndicators.RSI(df['Close'], 14)
                rsi = rsi_series.iloc[-1] if not rsi_series.dropna().empty else 50
                
                adx_series = TechnicalIndicators.ADX(df)
                adx = adx_series.iloc[-1] if not adx_series.dropna().empty else 0
                
                vol_change = 0.0
                if 'Volume' in df.columns and len(df) >= 5:
                    avg_volume = df['Volume'].iloc[-5:-1].mean()
                    if avg_volume > 0:
                        vol_change = (df['Volume'].iloc[-1] / avg_volume - 1) * 100
                
                market_data[name] = {
                    "ticker": ticker,
                    "last_close": float(last_close),
                    "returns_1d": float(returns_1d),
                    "returns_5d": float(returns_5d),
                    "rsi": float(rsi),
                    "adx": float(adx),
                    "volume_change": float(vol_change)
                }
            except Exception as e:
                logger.error(f"Erro ao analisar dados de mercado para {ticker}: {e}")
        
        return market_data

    @staticmethod
    def calculate_market_bias(market_data: dict) -> float:
        """Calcula o viés do mercado com base nos dados de referência."""
        if not market_data:
            return 0.0
        
        weights = {
            "US_MARKET": 0.40,
            "TECH_SECTOR": 0.20,
            "VOLATILITY": -0.15,
            "BONDS": -0.10,
            "GOLD": 0.05,
            "EMERGING": 0.05,
            "OIL": 0.05,
            "CRYPTO": 0.00
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for name, info in market_data.items():
            w = weights.get(name, 0)
            if w == 0:
                continue
            
            score = info.get("returns_1d", 0) * 2 + info.get("returns_5d", 0) * 0.5
            
            rsi = info.get("rsi", 50)
            if rsi > 70:
                score -= 20
            elif rsi < 30:
                score += 20
            else:
                score += (rsi - 50)
            
            if info.get("adx", 0) > 25:
                score *= 1.2
            
            total_weight += abs(w)
            adjusted_score = -score if w < 0 else score
            total_score += adjusted_score * abs(w)
        
        if total_weight == 0:
            return 0.0
        
        return float(np.clip(total_score / total_weight, -100, 100))

    @staticmethod
    def get_market_sentiment() -> Dict[str, Any]:
        """Obtém o sentimento geral do mercado."""
        data = MarketAnalysis.get_market_data()
        bias_score = MarketAnalysis.calculate_market_bias(data)
        
        if bias_score > 50:
            sentiment = "muito bullish"
        elif bias_score > 20:
            sentiment = "bullish"
        elif bias_score < -50:
            sentiment = "muito bearish"
        elif bias_score < -20:
            sentiment = "bearish"
        else:
            sentiment = "neutro"
        
        trend = "lateral"
        if "US_MARKET" in data:
            r5 = data["US_MARKET"].get("returns_5d", 0)
            if r5 > 2:
                trend = "alta"
            elif r5 < -2:
                trend = "baixa"
        
        return {
            "market_bias_score": bias_score,
            "sentiment": sentiment,
            "trend": trend,
            "details": data
        }