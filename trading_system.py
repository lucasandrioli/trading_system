#!/usr/bin/env python3
"""
Sistema de Trading Refatorado - Swing Trade Agressivo com yfinance
Este sistema utiliza análise técnica, qualitativa, fundamental, machine learning e gestão de risco
para fornecer recomendações de trading. Inclui estratégias de rebalanceamento inteligente
considerando o capital disponível, metas de recuperação e prazos.
"""

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Tuple, Any
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Create a session with connection pooling and retry logic
def create_session():
    """Create a session with retry logic and connection pooling"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Use this session for all HTTP requests
http_session = create_session()

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import time
import functools
import concurrent.futures
from functools import lru_cache

# Cache configuration
CACHE_ENABLED = True
CACHE_DURATION = 3600  # 1 hour in seconds

# Function cache decorator with time expiry
def timed_cache(seconds=CACHE_DURATION, maxsize=128):
    def wrapper_decorator(func):
        if not CACHE_ENABLED:
            return func
        
        @functools.lru_cache(maxsize=maxsize)
        def time_bounded(*args, _cache_time=None, **kwargs):
            if _cache_time is None or time.time() - _cache_time > seconds:
                _cache_time = time.time()
            return func(*args, **kwargs), _cache_time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result, _ = time_bounded(*args, _cache_time=int(time.time() / seconds), **kwargs)
            return result
        
        # Add a clear method
        wrapper.cache_clear = time_bounded.cache_clear
        return wrapper
    
    return wrapper_decorator

# Tenta importar tabulate para saída em tabela
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    tabulate = None
    HAS_TABULATE = False
    logging.warning("Biblioteca 'tabulate' não encontrada. A saída em tabela não estará disponível.")

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trading_system.log"), logging.StreamHandler()]
)
logger = logging.getLogger("trading_system")

# Parâmetros dinâmicos (indicadores, thresholds, pesos)
DYNAMIC_PARAMS = {
    "sma_period": 20,
    "ema_period": 9,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_window": 20,
    "bb_std": 2,
    "decision_buy_threshold": 60,
    "decision_buy_partial_threshold": 20,
    "decision_sell_threshold": -60,
    "decision_sell_partial_threshold": -20,
    "take_profit_pct": 5.0,
    "stop_loss_pct": -8.0,
    "trailing_stop_pct": 3.0,
    "weight_tech": 0.70,
    "weight_qual": 0.20,
    "weight_fund": 0.05,
    "weight_market": 0.05,
    "threshold_adjustment_factor": 0.25,
    "position_size_adjustment_max": 2.0,
    "sell_hysteresis_pct": 2.0,
    "buy_hysteresis_pct": 2.0,
    "rebalance_score_threshold": 40,  # Score mínimo para considerar rebalanceamento
    "min_sell_performance": -3.0,     # Piso de performance para considerar venda em rebalanceamento
    "recovery_urgency_factor": 0.15,  # Peso da urgência da meta de recuperação
    "min_transaction_value": 50.0,    # Valor mínimo para uma transação (para evitar comissões desnecessárias)
    "max_portfolio_assets": 15,       # Número máximo de ativos diferentes na carteira
    "min_risk_weight": 0.5            # Peso mínimo para risco (mesmo quando agressivo)
}

RISK_PROFILES = {
    "low": 0.8,
    "medium": 1.0,
    "high": 1.5,
    "ultra": 2.0
}

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

# Configuração da API Polygon (opcional)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or "YOUR_API_KEY_HERE"
if POLYGON_API_KEY == "YOUR_API_KEY_HERE":
    logger.info("Chave da Polygon não fornecida; prosseguindo sem Polygon para dados de mercado.")

# Ativa cache para requests (se disponível)
try:
    import requests_cache
    requests_cache.install_cache('yfinance_cache', backend='sqlite', expire_after=3600)
    logger.info("Cache de requests ativado (expire_after=3600s).")
    HAS_REQUESTS_CACHE = True
except ImportError:
    logger.warning("Biblioteca requests_cache não instalada. Recomenda-se instalar para melhor performance.")
    HAS_REQUESTS_CACHE = False


# =============================================================================
# 1. DataLoader – Obter dados via yfinance (ou Polygon se necessário)
# =============================================================================
class DataLoader:
    @staticmethod
    @timed_cache(seconds=300)  # 5-minute cache
    def get_asset_data(ticker: str, days: int = 60, interval: str = "1d",
                       use_polygon: bool = False, extended_hours: bool = False) -> pd.DataFrame:
        """Gets historical data for a specific ticker."""
        try:
            if not ticker or not isinstance(ticker, str):
                raise ValueError(f"Invalid ticker: {ticker}")
            
            # Try first with yfinance (more reliable)
            df = DataLoader._get_asset_data_yfinance(ticker, days, interval, include_prepost=extended_hours)
            
            # If yfinance fails and Polygon is configured, try Polygon as fallback
            if (df is None or df.empty) and use_polygon and POLYGON_API_KEY != "YOUR_API_KEY_HERE":
                logger.info(f"Data not found via yfinance for {ticker}, trying Polygon...")
                df = DataLoader._get_asset_data_polygon(ticker, days, interval)
            
            if df is None or df.empty:
                logger.warning(f"Empty data returned for {ticker}")
                return pd.DataFrame()
            
            # Optimize memory usage
            df = DataLoader._optimize_dataframe(df)
            
            return df
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage of dataframe"""
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        return df

    @staticmethod
    def get_realtime_prices_bulk(tickers: list) -> dict:
        """Gets real-time prices for multiple tickers using parallel processing."""
        prices = {}
        if not tickers:
            return prices
        
        # Split tickers into smaller batches
        max_batch_size = 30
        ticker_batches = [tickers[i:i + max_batch_size] for i in range(0, len(tickers), max_batch_size)]
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            batch_results = list(executor.map(DataLoader._get_batch_prices, ticker_batches))
            
        for result in batch_results:
            prices.update(result)
        
        # Process any missing tickers
        missing = [t for t in tickers if t not in prices]
        if missing:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(DataLoader._get_single_ticker_price, t): t for t in missing}
                for future in concurrent.futures.as_completed(futures):
                    ticker = futures[future]
                    try:
                        result = future.result()
                        if result:
                            prices[ticker] = result
                    except Exception as e:
                        logger.error(f"Error fetching price for {ticker}: {e}")
        
        return prices

    @staticmethod
    def _get_single_ticker_price(ticker):
        """Helper method for single ticker price"""
        try:
            hist = yf.Ticker(ticker).history(period="1d", auto_adjust=True)
            if not hist.empty and 'Close' in hist.columns:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass
        return None

    @staticmethod
    @lru_cache(maxsize=128)
    def check_ticker_valid(ticker: str) -> bool:
        """Cached version of ticker validation."""
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="5d")
            return not hist.empty
        except Exception:
            return False


# =============================================================================
# 2. TechnicalIndicators – Cálculo dos indicadores técnicos
# =============================================================================
class TechnicalIndicators:
    @staticmethod
    def SMA(series: pd.Series, window: int) -> pd.Series:
        """Calcula a Média Móvel Simples."""
        if series.empty or window <= 0:
            return pd.Series(index=series.index)
        return series.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def EMA(series: pd.Series, span: int) -> pd.Series:
        """Calcula a Média Móvel Exponencial."""
        if series.empty or span <= 0:
            return pd.Series(index=series.index)
        return series.ewm(span=span, adjust=False, min_periods=1).mean()

    @staticmethod
    def RSI(series: pd.Series, period: int = 14) -> pd.Series:
        """Calcula o Índice de Força Relativa (RSI)."""
        if series.empty or period <= 0:
            return pd.Series(index=series.index)
        
        delta = series.diff().dropna()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = pd.Series(index=avg_gain.index, dtype=float)
        valid_idx = avg_loss > 0
        rs[valid_idx] = avg_gain[valid_idx] / avg_loss[valid_idx]
        rs[~valid_idx & (avg_gain > 0)] = 100.0
        rs[~valid_idx & (avg_gain == 0)] = 0.0
        
        rsi = 100 - (100 / (1 + rs))
        rsi[~valid_idx & (avg_gain == 0)] = 50.0
        
        return rsi

    @staticmethod
    def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calcula o MACD (Moving Average Convergence Divergence)."""
        if series.empty or fast <= 0 or slow <= 0 or signal <= 0:
            empty_series = pd.Series(index=series.index)
            return empty_series, empty_series, empty_series
        
        ema_fast = TechnicalIndicators.EMA(series, fast)
        ema_slow = TechnicalIndicators.EMA(series, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = TechnicalIndicators.EMA(macd_line, signal)
        macd_hist = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_hist

    @staticmethod
    def BollingerBands(series: pd.Series, window: int = 20, num_std: float = 2):
        """Calcula as Bandas de Bollinger."""
        if series.empty or window <= 0:
            empty_series = pd.Series(index=series.index)
            return empty_series, empty_series, empty_series
        
        sma = TechnicalIndicators.SMA(series, window)
        std = series.rolling(window=window, min_periods=1).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return sma, upper_band, lower_band

    @staticmethod
    def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula o Average True Range."""
        if df.empty or period <= 0 or 'High' not in df.columns or 'Low' not in df.columns:
            return pd.Series(index=df.index)
        
        df = df.copy()
        if 'Close' not in df.columns:
            logger.warning("Coluna 'Close' ausente ao calcular ATR")
            df['Close'] = df['High'] - (df['High'] - df['Low']) / 2
        
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        tr = ranges.max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr

    @staticmethod
    def ADX(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula o Average Directional Index."""
        if df.empty or period <= 0 or 'High' not in df.columns or 'Low' not in df.columns:
            return pd.Series(index=df.index)
        
        dfc = df.copy()
        if 'Close' not in dfc.columns:
            logger.warning("Coluna 'Close' ausente ao calcular ADX")
            dfc['Close'] = dfc['High'] - (dfc['High'] - dfc['Low']) / 2
        
        dfc['TR'] = np.maximum(
            dfc['High'] - dfc['Low'],
            np.maximum(
                np.abs(dfc['High'] - dfc['Close'].shift(1, fill_value=dfc['High'].iloc[0])),
                np.abs(dfc['Low'] - dfc['Close'].shift(1, fill_value=dfc['Low'].iloc[0]))
            )
        )
        
        dfc['+DM'] = np.where(
            (dfc['High'] - dfc['High'].shift(1)) > (dfc['Low'].shift(1) - dfc['Low']),
            np.maximum(dfc['High'] - dfc['High'].shift(1), 0),
            0
        )
        
        dfc['-DM'] = np.where(
            (dfc['Low'].shift(1) - dfc['Low']) > (dfc['High'] - dfc['High'].shift(1)),
            np.maximum(dfc['Low'].shift(1) - dfc['Low'], 0),
            0
        )
        
        atr = dfc['TR'].rolling(window=period, min_periods=1).mean()
        plus_di_raw = dfc['+DM'].rolling(window=period, min_periods=1).sum()
        minus_di_raw = dfc['-DM'].rolling(window=period, min_periods=1).sum()
        
        non_zero_atr = atr.replace(0, np.nan)
        dfc['+DI'] = 100 * (plus_di_raw / non_zero_atr)
        dfc['-DI'] = 100 * (minus_di_raw / non_zero_atr)
        
        sum_di = dfc['+DI'] + dfc['-DI']
        diff_di = (dfc['+DI'] - dfc['-DI']).abs()
        
        dx = pd.Series(index=dfc.index, dtype=float)
        non_zero_sum = sum_di > 0
        dx[non_zero_sum] = 100 * (diff_di[non_zero_sum] / sum_di[non_zero_sum])
        
        adx = dx.rolling(window=period, min_periods=1).mean()
        return adx

    @staticmethod
    def stochastic_oscillator(df: pd.DataFrame, period: int = 14):
        """Calcula o Oscilador Estocástico."""
        if df.empty or period <= 0 or not set(['High', 'Low', 'Close']).issubset(df.columns):
            empty_series = pd.Series(index=df.index)
            return empty_series, empty_series
        
        lowest_low = df['Low'].rolling(window=period, min_periods=1).min()
        highest_high = df['High'].rolling(window=period, min_periods=1).max()
        range_diff = highest_high - lowest_low
        
        percent_k = pd.Series(index=df.index, dtype=float)
        non_zero_range = range_diff > 0
        percent_k[non_zero_range] = 100 * ((df['Close'][non_zero_range] - lowest_low[non_zero_range]) / range_diff[non_zero_range])
        percent_k[~non_zero_range] = 50.0
        
        percent_d = percent_k.rolling(window=3, min_periods=1).mean()
        return percent_k, percent_d

    @staticmethod
    def ichimoku_cloud(df: pd.DataFrame, conv_period: int = 9, base_period: int = 26,
                       span_b_period: int = 52, displacement: int = 26):
        """Calcula a Nuvem de Ichimoku."""
        if df.empty or not set(['High', 'Low']).issubset(df.columns):
            empty_series = pd.Series(index=df.index)
            return empty_series, empty_series, empty_series, empty_series
        
        high = df['High']
        low = df['Low']
        
        conv_high = high.rolling(window=conv_period, min_periods=1).max()
        conv_low = low.rolling(window=conv_period, min_periods=1).min()
        conv_line = (conv_high + conv_low) / 2
        
        base_high = high.rolling(window=base_period, min_periods=1).max()
        base_low = low.rolling(window=base_period, min_periods=1).min()
        base_line = (base_high + base_low) / 2
        
        span_a = ((conv_line + base_line) / 2).shift(displacement, fill_value=np.nan)
        
        span_b_high = high.rolling(window=span_b_period, min_periods=1).max()
        span_b_low = low.rolling(window=span_b_period, min_periods=1).min()
        span_b = ((span_b_high + span_b_low) / 2).shift(displacement, fill_value=np.nan)
        
        return conv_line, base_line, span_a, span_b

    @staticmethod
    def OBV(df: pd.DataFrame) -> pd.Series:
        """Calcula o On-Balance Volume."""
        if df.empty or not set(['Close', 'Volume']).issubset(df.columns):
            return pd.Series(index=df.index)
        
        obv = pd.Series(0, index=df.index, dtype='float64')
        if len(df) <= 1:
            return obv
        
        for i in range(1, len(df)):
            if pd.isna(df['Close'].iloc[i]) or pd.isna(df['Close'].iloc[i-1]) or pd.isna(df['Volume'].iloc[i]):
                obv.iloc[i] = obv.iloc[i-1]
            elif df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

    @staticmethod
    def add_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Adiciona todos os indicadores técnicos ao DataFrame."""
        if df.empty or 'Close' not in df.columns:
            logger.warning("DataFrame vazio ou sem coluna 'Close' ao adicionar indicadores")
            return df
        
        df_ind = df.copy()
        try:
            df_ind['SMA'] = TechnicalIndicators.SMA(df_ind['Close'], params.get("sma_period", 20))
            df_ind['EMA'] = TechnicalIndicators.EMA(df_ind['Close'], params.get("ema_period", 9))
            df_ind['RSI'] = TechnicalIndicators.RSI(df_ind['Close'], params.get("rsi_period", 14))
            
            macd_line, macd_signal, macd_hist = TechnicalIndicators.MACD(
                df_ind['Close'],
                params.get("macd_fast", 12),
                params.get("macd_slow", 26),
                params.get("macd_signal", 9)
            )
            df_ind['MACD_line'] = macd_line
            df_ind['MACD_signal'] = macd_signal
            df_ind['MACD_hist'] = macd_hist
            
            bb_mid, bb_upper, bb_lower = TechnicalIndicators.BollingerBands(
                df_ind['Close'],
                params.get("bb_window", 20),
                params.get("bb_std", 2)
            )
            df_ind['BB_mid'] = bb_mid
            df_ind['BB_upper'] = bb_upper
            df_ind['BB_lower'] = bb_lower
            
            if set(['High', 'Low']).issubset(df_ind.columns):
                df_ind['ATR'] = TechnicalIndicators.ATR(df_ind)
                df_ind['ADX'] = TechnicalIndicators.ADX(df_ind)
                df_ind['%K'], df_ind['%D'] = TechnicalIndicators.stochastic_oscillator(df_ind)
                df_ind['Ichimoku_conv'], df_ind['Ichimoku_base'], df_ind['Ichimoku_spanA'], df_ind['Ichimoku_spanB'] = TechnicalIndicators.ichimoku_cloud(df_ind)
            
            if 'Volume' in df_ind.columns:
                df_ind['OBV'] = TechnicalIndicators.OBV(df_ind)
                df_ind['OBV_SMA'] = TechnicalIndicators.SMA(df_ind['OBV'], 10)
        
        except Exception as e:
            logger.error(f"Erro ao adicionar indicadores técnicos: {e}")
        
        return df_ind


# =============================================================================
# 3. MarketAnalysis – Análise do sentimento do mercado
# =============================================================================
class MarketAnalysis:
    @staticmethod
    def get_market_data() -> dict:
        """Obtém dados de mercado para referências-chave (índices, setores, etc.)."""
        market_data = {}
        
        for name, ticker in MARKET_REFERENCE_TICKERS.items():
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
    def get_market_sentiment() -> dict:
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


# =============================================================================
# 4. APIClient – Funções para Polygon (notícias e relacionados)
# =============================================================================
class APIClient:
    @staticmethod
    def get_polygon_technical_data(ticker: str, indicator: str, params: dict = None) -> dict:
        """Gets technical indicator data via Polygon API."""
        if not POLYGON_API_KEY or POLYGON_API_KEY == "YOUR_API_KEY_HERE":
            return {}
        
        try:
            url = f"https://api.polygon.io/v1/indicators/{indicator}/{ticker}"
            req_params = {"apiKey": POLYGON_API_KEY}
            if params:
                req_params.update(params)
                
            resp = http_session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", {})
        except Exception as e:
            logger.error(f"Error getting technical data from Polygon: {e}")
            return {}
    
    @staticmethod
    def get_polygon_related(ticker: str) -> list:
        """Gets related tickers via Polygon API."""
        # Keep this method as is - it doesn't fetch news
        if not POLYGON_API_KEY or POLYGON_API_KEY == "YOUR_API_KEY_HERE":
            return []
        
        try:
            url = f"https://api.polygon.io/vX/reference/tickers/{ticker}/relationships"
            params = {"apiKey": POLYGON_API_KEY}
            resp = http_session.get(url, params=params, timeout=10)
            
            if resp.status_code == 404:
                url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
                resp = http_session.get(url, params=params, timeout=10)
            
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            results = []
            
            if "results" in data:
                if "related_tickers" in data["results"]:
                    results = data["results"]["related_tickers"]
                elif "brands" in data["results"]:
                    for brand in data["results"]["brands"]:
                        results.extend(brand.get("tickers", []))
            
            return list(set(results))
        except Exception as e:
            logger.error(f"Error getting related tickers from Polygon for {ticker}: {e}")
            return []


# =============================================================================
# 5. QualitativeAnalysis – Análise de notícias e sentimento
# =============================================================================
class QualitativeAnalysis:
    @staticmethod
    def analyze_news_sentiment(ticker: str) -> dict:
        """Analyzes sentiment of news related to a ticker using only yfinance."""
        news_items = []
        
        # Use ONLY yfinance for news
        try:
            yf_news = yf.Ticker(ticker).get_news()
            if yf_news:
                for item in yf_news:
                    news_items.append({
                        "title": item.get("title", ""),
                        "published": item.get("publisher"),
                        "time": item.get("providerPublishTime", None),
                        "url": item.get("link", "")
                    })
            else:
                logger.info(f"No news found via yfinance for {ticker}.")
        except Exception as e:
            logger.error(f"Error fetching news via yfinance: {e}")
        
        # Remove duplicates
        titles_seen = set()
        unique_news = []
        for item in news_items:
            title = item.get("title", "")
            if title and title not in titles_seen:
                titles_seen.add(title)
                unique_news.append(item)
        
        news_items = unique_news
        
        # Calculate news sentiment
        total_weighted_sent = 0.0
        total_weight = 0.0
        sentiment_details = []
        now = datetime.now(timezone.utc)
        
        for item in news_items:
            title = item.get("title", "")
            if not title:
                continue
            
            try:
                if item.get("published"):
                    if isinstance(item["published"], str):
                        pub_date = datetime.strptime(item["published"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    else:
                        pub_date = datetime.fromtimestamp(item["time"], tz=timezone.utc)
                else:
                    pub_date = now
            except Exception:
                pub_date = now
            
            days_ago = max(0, (now - pub_date).days)
            weight = 1 / (1 + 0.5 * days_ago)
            
            polarity = TextBlob(title).sentiment.polarity
            if abs(polarity) > 0.5:
                polarity *= 1.2
            
            total_weighted_sent += polarity * weight
            total_weight += weight
            
            sentiment_details.append({
                "title": title,
                "date": pub_date.strftime("%Y-%m-%d"),
                "sentiment": float(polarity),
                "weight": round(weight, 2)
            })
        
        # Calculate average sentiment
        avg_sentiment = total_weighted_sent / total_weight if total_weight > 0 else 0.0
        
        if avg_sentiment > 0.3:
            sentiment_label = "muito positivo"
        elif avg_sentiment > 0:
            sentiment_label = "positivo"
        elif avg_sentiment < -0.3:
            sentiment_label = "muito negativo"
        elif avg_sentiment < 0:
            sentiment_label = "negativo"
        else:
            sentiment_label = "neutro"
        
        return {
            "sentiment_score": float(avg_sentiment),
            "sentiment_label": sentiment_label,
            "news_count": len(news_items),
            "details": sentiment_details
        }

# =============================================================================
# 6. FundamentalAnalysis – Análise fundamental com yfinance
# =============================================================================
class FundamentalAnalysis:
    @staticmethod
    def fundamental_score(ticker: str) -> dict:
        """Calcula um score fundamental para o ticker com base em dados do yfinance."""
        score = 0
        details = {}
        
        try:
            ticker_obj = yf.Ticker(ticker)
            
            try:
                info = ticker_obj.info or {}
            except Exception as e:
                logger.warning(f"Erro ao obter info para {ticker}: {e}")
                info = {}
            
            try:
                fast_info = ticker_obj.get_fast_info() or {}
            except Exception as e:
                logger.warning(f"Erro ao obter fast_info para {ticker}: {e}")
                fast_info = {}
            
            try:
                isin = ticker_obj.get_isin()
            except Exception:
                isin = None
            
            # Coleta outros dados disponíveis
            try:
                dividends = ticker_obj.get_dividends()
            except Exception:
                dividends = pd.Series()
            
            try:
                splits = ticker_obj.get_splits()
            except Exception:
                splits = pd.Series()
            
            try:
                actions = ticker_obj.get_actions()
            except Exception:
                actions = pd.Series()
            
            try:
                capital_gains = ticker_obj.get_capital_gains()
            except Exception:
                capital_gains = pd.Series()
            
            try:
                history_metadata = ticker_obj.get_history_metadata()
            except Exception:
                history_metadata = {}
            
            try:
                news = ticker_obj.get_news()
            except Exception:
                news = []
            
            # Armazena dados coletados
            details["isin"] = isin
            details["fast_info"] = {k: v for k, v in fast_info.items() if isinstance(v, (int, float, str, bool))}
            details["history_metadata"] = history_metadata
            details["dividends"] = dividends.to_dict() if not dividends.empty else {}
            details["splits"] = splits.to_dict() if not splits.empty else {}
            details["actions"] = actions.to_dict() if hasattr(actions, "to_dict") else {}
            details["capital_gains"] = capital_gains.to_dict() if hasattr(capital_gains, "to_dict") else {}
            details["news_count_yf"] = len(news) if news else 0
            
            # Análise específica por tipo de ativo
            quote_type = info.get("quoteType", "").upper()
            
            if quote_type in ["ETF", "MUTUALFUND"]:
                ytd = info.get("ytdReturn", 0) or 0
                
                try:
                    spy_ytd = yf.Ticker("SPY").info.get("ytdReturn", 0) or 0
                except Exception:
                    spy_ytd = 0
                
                rel_perf = ytd - spy_ytd
                perf_score = np.clip(rel_perf * 100, -25, 25)
                score += perf_score
                
                details["ytd_return"] = ytd
                details["rel_perf_vs_SPY"] = rel_perf
                
                short_name = info.get("shortName", "").upper()
                if any(term in short_name for term in ["SHORT", "INVERSE", "BEAR"]):
                    inv_adj = -10 if spy_ytd > 0 else 10
                    score += inv_adj
                    details["inverse_fund_adjustment"] = inv_adj
            
            elif quote_type in ["EQUITY", "STOCK"]:
                # Análise do P/E
                pe = info.get("trailingPE") or info.get("forwardPE")
                if pe:
                    if pe < 15:
                        score += 10
                        details["PE"] = "baixo"
                    elif pe > 50:
                        score -= 5
                        details["PE"] = "alto"
                    details["pe_ratio"] = pe
                
                # Crescimento da receita
                rg = info.get("revenueGrowth")
                if rg:
                    rg_pct = rg * 100
                    rev_score = np.clip(rg_pct, -20, 20)
                    score += rev_score
                    details["revenue_growth_pct"] = rg_pct
                
                # Margens de lucro
                pm = info.get("profitMargins")
                if pm is not None:
                    if pm > 0.2:
                        score += 15
                    elif pm > 0.1:
                        score += 10
                    elif pm > 0:
                        score += 5
                    else:
                        score -= 10
                    details["profit_margin_pct"] = pm * 100
                
                # Dívida sobre patrimônio
                dte = info.get("debtToEquity")
                if dte:
                    if dte > 2:
                        score -= 15
                    elif dte > 1:
                        score -= 5
                    else:
                        score += 5
                    details["debt_to_equity"] = dte
                
                # Dividendos
                dividend_yield = info.get("dividendYield")
                details["dividend_yield"] = dividend_yield
                if dividend_yield is not None:
                    score += 5 if dividend_yield > 0.03 else 2
                
                # Splits (penaliza por diluição)
                num_splits = len(splits) if not splits.empty else 0
                details["num_splits"] = num_splits
                if num_splits > 0:
                    score -= min(num_splits, 5)
            
            else:
                details["note"] = f"Tipo de ativo não suportado para análise fundamental: {quote_type}"
            
            details["raw_score"] = score
            return {"fundamental_score": float(np.clip(score, -100, 100)), "details": details}
        
        except Exception as e:
            logger.error(f"Erro ao obter info fundamental de {ticker}: {e}")
            return {"fundamental_score": 0, "details": {"error": str(e)}}


# =============================================================================
# 7. Strategy – Combinação dos scores e decisão final de trade
# =============================================================================
class Strategy:
    @staticmethod
    def technical_score(df: pd.DataFrame, risk_profile: str = "medium", params: dict = DYNAMIC_PARAMS) -> dict:
        """Calcula o score técnico com base nos indicadores do DataFrame."""
        if df.empty or 'Close' not in df.columns or len(df) < 2:
            return {"score": 0.0, "details": {"error": "Dados insuficientes para análise técnica"}}
        
        last = df.iloc[-1]
        score = 0.0
        details = {}
        
        # RSI
        if 'RSI' in last:
            rsi = last['RSI'] if not pd.isna(last['RSI']) else 50
            details['RSI'] = float(rsi)
            
            if rsi <= 30:
                rsi_score = 25 * (1 - rsi/30)
                details['RSI_signal'] = "sobrevendido"
            elif rsi >= 70:
                rsi_score = -25 * ((rsi - 70) / 30)
                details['RSI_signal'] = "sobrecomprado"
            else:
                rsi_score = (50 - rsi) / 2
                details['RSI_signal'] = "neutro"
            
            score += rsi_score
            details['rsi_score'] = float(rsi_score)
        else:
            details['RSI'] = None
            details['rsi_score'] = 0.0
        
        # MACD
        macd_hist = last.get('MACD_hist', 0) if not pd.isna(last.get('MACD_hist', 0)) else 0
        macd_score = float(np.clip(60 * macd_hist, -25, 25))
        score += macd_score
        details['MACD_hist'] = float(macd_hist)
        details['macd_score'] = float(macd_score)
        
        # ADX
        adx = last.get('ADX', 0) if not pd.isna(last.get('ADX', 0)) else 0
        details['ADX'] = float(adx)
        
        if adx > 30:
            adx_score = 15
        elif adx > 25:
            adx_score = 10
        elif adx > 20:
            adx_score = 5
        else:
            adx_score = 0
        
        score += adx_score
        details['adx_score'] = adx_score
        
        # Bollinger Bands
        current_price = last['Close']
        bb_lower = last.get('BB_lower')
        bb_upper = last.get('BB_upper')
        
        if bb_lower is not None and bb_upper is not None and not pd.isna(bb_lower) and not pd.isna(bb_upper):
            if current_price <= bb_lower:
                bb_score = 15
                details['bb_signal'] = "abaixo da banda inferior"
            elif current_price >= bb_upper:
                bb_score = -15
                details['bb_signal'] = "acima da banda superior"
            else:
                bb_range = bb_upper - bb_lower if bb_upper - bb_lower > 0 else 1
                rel_pos = (current_price - bb_lower) / bb_range
                
                if rel_pos < 0.3:
                    bb_score = 10
                    details['bb_signal'] = "próximo da banda inferior"
                elif rel_pos > 0.7:
                    bb_score = -10
                    details['bb_signal'] = "próximo da banda superior"
                else:
                    bb_score = 0
                    details['bb_signal'] = "no centro das bandas"
                
                details['bb_position_pct'] = round(rel_pos * 100, 1)
            
            score += bb_score
            details['bb_score'] = bb_score
        else:
            details['bb_score'] = 0
        
        # Previsão via ML
        ml_pred = ml_price_forecast(df)
        if ml_pred:
            pct_change = ml_pred["change_pct"]
            forecast_score = float(np.clip(pct_change * 0.6, -15, 15))
            score += forecast_score
            details['forecast_pct_change'] = round(pct_change, 2)
            details['forecast_score'] = float(forecast_score)
        
        # OBV Score
        obv_val = obv_score(df)
        score += obv_val
        details['obv_score'] = float(obv_val)
        
        # Estocástico
        stoch_k = last.get('%K', 50)
        stoch_d = last.get('%D', 50)
        
        if not pd.isna(stoch_k) and not pd.isna(stoch_d):
            if stoch_k < 20 and stoch_d < 20:
                stoch_score = 10
            elif stoch_k > 80 and stoch_d > 80:
                stoch_score = -10
            elif stoch_k > stoch_d and stoch_k < 50:
                stoch_score = 8
            elif stoch_k < stoch_d and stoch_k > 50:
                stoch_score = -8
            else:
                stoch_score = 0
            
            score += stoch_score
            details['stochastic_score'] = stoch_score
        else:
            details['stochastic_score'] = 0
        
        # Tendência Recente
        if len(df) >= 6:
            recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
            
            if abs(recent_return) > 5:
                rsi_val = details.get('RSI', 50)
                trend_score = 10 if (recent_return > 0 and rsi_val < 60) else (-10 if recent_return < 0 and rsi_val > 40 else 5 if recent_return > 0 else -5)
            else:
                trend_score = 0
            
            score += trend_score
            details['trend_score'] = trend_score
            details['recent_return_5d_pct'] = round(recent_return, 2)
        
        # Ajuste pelo perfil de risco
        risk_mult = RISK_PROFILES.get(risk_profile, 1.0)
        score *= risk_mult
        
        details['risk_profile'] = risk_profile
        details['risk_multiplier'] = risk_mult
        
        final_score = float(np.clip(score, -100, 100))
        details['raw_score'] = round(score, 2)
        
        return {"score": final_score, "details": details}

    @staticmethod
    def market_adaptive_score(ticker: str, market_sentiment: dict) -> float:
        """Calcula um score adaptativo de mercado para o ticker."""
        try:
            info = yf.Ticker(ticker).info
            beta = info.get("beta", 1.0)
            
            if beta is None or pd.isna(beta):
                beta = 1.0
            
            quote_type = info.get("quoteType", "")
            market_bias = market_sentiment.get("market_bias_score", 0)
            
            # Identificar ETFs inversos
            is_inverse = (quote_type == "ETF") and any(term in info.get("shortName", "").upper() for term in ["SHORT", "INVERSE", "BEAR"])
            
            # Para ETFs inversos, o score é o oposto do mercado
            adaptive_score = -market_bias if is_inverse else market_bias
            
            # Ajuste pelo beta (maior beta = mais sensível ao mercado)
            if beta > 0:
                adaptive_score *= min(beta, 2.0)
            
            return float(np.clip(adaptive_score, -100, 100))
        
        except Exception as e:
            logger.error(f"Erro ao calcular market_adaptive_score para {ticker}: {e}")
            return 0.0

    @staticmethod
    def calculate_recovery_urgency(goals: dict, remaining_days: int = None) -> float:
        """Calcula um fator de urgência baseado na meta de recuperação e dias restantes."""
        if not goals:
            return 0.0
        
        target_recovery = goals.get('target_recovery', 0)
        total_days = goals.get('days', 0)
        
        if target_recovery <= 0 or total_days <= 0:
            return 0.0
        
        # Se não for especificado, assume que hoje é o primeiro dia
        if remaining_days is None:
            remaining_days = total_days
        
        # Limita remaining_days para evitar divisão por zero ou valores negativos
        remaining_days = max(1, min(remaining_days, total_days))
        
        # Calcula o progresso do tempo (0 = início, 1 = fim)
        time_progress = 1 - (remaining_days / total_days)
        
        # A urgência aumenta à medida que o tempo passa
        # Começa em 0.5 e vai até 2.0 nos últimos dias
        urgency_factor = 0.5 + (1.5 * (time_progress ** 2))
        
        return float(urgency_factor)

    @staticmethod
    def decision_engine(ticker: str, df: pd.DataFrame, position: dict, account_balance: float,
                        risk_profile: str, params: dict, daily_gap: float = 0.0, daily_goal: float = 0.0,
                        market_sentiment: dict = None, trailing_data: dict = None, 
                        goals: dict = None, remaining_days: int = None) -> dict:
        """Motor de decisão combinando todos os fatores para decidir ações de trading."""
        if df is None or df.empty or len(df) < 20:
            return {
                "ticker": ticker,
                "decision": "AGUARDAR",
                "confidence": "BAIXA",
                "justificativa": "Dados insuficientes para análise.",
                "quantity": position.get("quantity", 0)
            }
        
        # Adicionar indicadores técnicos ao DataFrame
        df_ind = TechnicalIndicators.add_all_indicators(df, params)
        current_price = float(df_ind['Close'].iloc[-1])
        
        if np.isnan(current_price):
            return {
                "ticker": ticker,
                "decision": "AGUARDAR",
                "justificativa": "Preço atual indisponível",
                "quantity": position.get("quantity", 0)
            }
        
        # Calcular scores técnico, qualitativo, fundamental e de mercado
        tech_result = Strategy.technical_score(df_ind, risk_profile, params)
        technical_score = tech_result["score"]
        
        qual_result = QualitativeAnalysis.qualitative_score(ticker)
        qualitative_score = qual_result["qualitative_score"]
        
        fund_result = FundamentalAnalysis.fundamental_score(ticker)
        fundamental_score = fund_result.get("fundamental_score", 0)
        
        market_score = 0.0
        if market_sentiment:
            market_score = Strategy.market_adaptive_score(ticker, market_sentiment)
        
        # Calcular score total ponderado
        total_score = (
            params["weight_tech"] * technical_score +
            params["weight_qual"] * qualitative_score +
            params["weight_fund"] * fundamental_score +
            params["weight_market"] * market_score
        )
        total_score = float(np.clip(total_score, -100, 100))
        
        # Ajustar thresholds baseado na meta de recuperação e tempo restante
        recovery_urgency = 1.0
        if goals and remaining_days is not None:
            recovery_urgency = Strategy.calculate_recovery_urgency(goals, remaining_days)
        
        gap_ratio = 0
        if daily_goal > 0:
            gap_ratio = min(max(daily_gap / daily_goal, 0), 1.0)
        
        adj_factor = params.get("threshold_adjustment_factor", 0.25) * recovery_urgency
        
        # Ajustar thresholds de decisão
        buy_threshold = params["decision_buy_threshold"] * (1 - adj_factor * gap_ratio)
        buy_partial_threshold = params["decision_buy_partial_threshold"] * (1 - adj_factor * gap_ratio)
        sell_threshold = params["decision_sell_threshold"] * (1 + adj_factor * gap_ratio)
        sell_partial_threshold = params["decision_sell_partial_threshold"] * (1 + adj_factor * gap_ratio)
        
        # Informações da posição atual
        quantity = position.get("quantity", 0)
        avg_price = position.get("avg_price", 0)
        position_exists = quantity > 0 and avg_price > 0
        position_profit_pct = ((current_price - avg_price) / avg_price * 100) if position_exists else 0.0
        position_profit_value = (current_price - avg_price) * quantity if position_exists else 0.0
        
        # Inicialização da decisão
        decision = "AGUARDAR"
        justification = "Sinal neutro ou insuficiente"
        trailing_triggered = False
        trailing_info = None
        
        # Verificar trailing stop
        if trailing_data is not None and position_exists:
            result_ts = check_trailing_stop(ticker, position, current_price, trailing_data, params)
            trailing_info = result_ts
            if result_ts.get("triggered"):
                decision = "VENDER"
                justification = result_ts.get("reason", "Trailing stop atingido.")
                trailing_triggered = True
        
        # Lógica de decisão
        if not trailing_triggered:
            # Verificar stop loss
            if position_exists and position_profit_pct <= params.get("stop_loss_pct", -8.0):
                decision = "VENDER"
                justification = f"Stop Loss atingido ({position_profit_pct:.2f}%)"
            
            # Verificar take profit
            elif position_exists and position_profit_pct >= params.get("take_profit_pct", 5.0):
                if total_score < 0:
                    decision = "VENDER"
                    justification = f"Take Profit de {position_profit_pct:.2f}% atingido e sinais enfraquecendo"
                else:
                    decision = "REALIZAR LUCRO PARCIAL"
                    justification = f"Take Profit de {position_profit_pct:.2f}% atingido; manter parcialmente"
            
            # Verificar score de compra
            elif total_score >= buy_threshold:
                decision = "COMPRAR"
                justification = "Score muito positivo"
            
            # Verificar score de compra parcial
            elif total_score >= buy_partial_threshold:
                decision = "COMPRAR PARCIAL"
                justification = "Score moderadamente positivo"
            
            # Verificar score de venda
            elif total_score <= sell_threshold and position_exists:
                decision = "VENDER"
                justification = "Score muito negativo"
            
            # Verificar score de venda parcial
            elif total_score <= sell_partial_threshold and position_exists:
                if position_profit_pct > 0:
                    decision = "REALIZAR LUCRO PARCIAL"
                    justification = "Tendência negativa após lucro acumulado"
                else:
                    decision = "REDUZIR"
                    justification = "Sinais negativos - reduzir posição"
            
            # Caso padrão
            else:
                decision = "AGUARDAR"
                justification = "Sem sinais claros o suficiente"
            
            # Verificar histerese de venda
            if decision in ["VENDER", "REALIZAR LUCRO PARCIAL", "REDUZIR"]:
                last_sell = position.get("last_sell")
                if last_sell:
                    hyst = params.get("sell_hysteresis_pct", 2.0)
                    prev_price = last_sell.get("price", current_price)
                    if current_price > prev_price * (1 - hyst/100):
                        decision = "AGUARDAR"
                        justification += " | Histerese: preço não caiu o suficiente desde última venda."
            
            # Verificar histerese de compra
            if decision in ["COMPRAR", "COMPRAR PARCIAL"]:
                last_buy = position.get("last_buy")
                if last_buy:
                    hyst = params.get("buy_hysteresis_pct", 2.0)
                    prev_price = last_buy.get("price", current_price)
                    if current_price < prev_price * (1 + hyst/100):
                        decision = "AGUARDAR"
                        justification += " | Histerese: preço não subiu o suficiente desde última compra."
        
        # Construir resultado
        result = {
            "ticker": ticker,
            "current_price": current_price,
            "total_score": total_score,
            "technical_score": technical_score,
            "qualitative_score": qualitative_score,
            "fundamental_score": fundamental_score,
            "market_score": market_score,
            "position_profit_pct": position_profit_pct,
            "position_profit_value": position_profit_value,
            "decision": decision,
            "justificativa": justification,
            "details": {
                "technical": tech_result["details"],
                "qualitative": qual_result.get("details", {}),
                "fundamental": fund_result.get("details", {}),
                "trailing_stop": trailing_info,
                "recovery_urgency": recovery_urgency
            },
            "thresholds": {
                "buy": buy_threshold,
                "buy_partial": buy_partial_threshold,
                "sell": sell_threshold,
                "sell_partial": sell_partial_threshold
            },
            "quantity": quantity  # Quantidade na carteira
        }
        
        # Para compras, calcula o sizing e inclui a quantidade recomendada
        if decision.upper() in ["COMPRAR", "COMPRAR PARCIAL"]:
            pos_size = PositionSizing.calculate_position_size(
                ticker, df, account_balance, risk_profile,
                daily_gap=daily_gap, daily_goal=daily_goal, params=params
            )
            result["position_sizing"] = pos_size
            result["recommended_quantity"] = pos_size.get("suggested_shares", 0)
        
        # Para vendas, calcula a quantidade recomendada de venda
        if decision.upper() in ["VENDER", "REALIZAR LUCRO PARCIAL", "REDUZIR"] and position_exists:
            sell_qty = quantity if decision.upper() == "VENDER" else max(1, int(quantity * 0.5))
            
            # Se já houver recomendação específica, use-a
            if result.get("sell_recommendation"):
                sell_qty = result["sell_recommendation"].get("sell_quantity", sell_qty)
            else:
                result["sell_recommendation"] = {
                    "sell_quantity": sell_qty,
                    "estimated_value": sell_qty * current_price
                }
            
            result["recommended_quantity"] = sell_qty
        
        return result


# =============================================================================
# Funções auxiliares: Previsão ML, OBV e trailing stop
# =============================================================================
def ml_price_forecast(df: pd.DataFrame, days_ahead: int = 1) -> Optional[Dict]:
    """Realiza previsão de preço usando ML para o próximo período."""
    try:
        if df is None or df.empty or len(df) < 5 or 'Close' not in df.columns:
            return None
        
        df_temp = df.reset_index()
        
        if 'Date' in df_temp.columns:
            df_temp['DayCount'] = (df_temp['Date'] - df_temp['Date'].min()).dt.days
        else:
            df_temp['DayCount'] = np.arange(len(df_temp))
        
        X = df_temp[['DayCount']]
        y = df_temp['Close']
        
        # Escolhe o modelo com base no tamanho dos dados
        model = RandomForestRegressor(n_estimators=100, random_state=42) if len(df_temp) >= 10 else LinearRegression()
        model.fit(X, y)
        
        next_day = df_temp['DayCount'].max() + days_ahead
        
        # Use a DataFrame with feature names for prediction to avoid the warning
        pred_df = pd.DataFrame([[next_day]], columns=['DayCount'])
        pred_price = model.predict(pred_df)[0]
        
        current_price = float(y.iloc[-1])
        change_pct = (pred_price - current_price) / current_price * 100
        confidence = model.score(X, y)
        
        return {
            "current_price": current_price,
            "predicted_price": float(pred_price),
            "change_pct": float(change_pct),
            "confidence": confidence,
            "model_r2": confidence
        }
    except Exception as e:
        logger.error(f"Erro na previsão de preço via ML: {e}")
        return None

def obv_score(df: pd.DataFrame) -> float:
    """Calcula um score baseado no On-Balance Volume (OBV)."""
    try:
        if df is None or df.empty or 'OBV' not in df.columns or 'Close' not in df.columns:
            return 0.0
        
        obv_series = df['OBV']
        price_series = df['Close']
        
        if len(obv_series) < 6:
            return 0.0
        
        # Calcular mudanças de preço e OBV
        price_change = price_series.iloc[-1] / price_series.iloc[-6] - 1
        denom = abs(obv_series.iloc[-6]) + 1e-9  # Evitar divisão por zero
        obv_change = (obv_series.iloc[-1] - obv_series.iloc[-6]) / denom
        
        score = 0.0
        
        # Avaliar concordância/divergência entre preço e OBV
        if price_change > 0 and obv_change > 0:
            score += 5.0 * min(1.0, obv_change * 10)
        elif price_change < 0 and obv_change < 0:
            score -= 5.0 * min(1.0, abs(obv_change) * 10)
        elif price_change > 0 and obv_change < 0:
            score -= 3.0  # Divergência negativa
        elif price_change < 0 and obv_change > 0:
            score += 3.0  # Divergência positiva
        
        # Verificar posição em relação à média móvel
        obv_ma = obv_series.rolling(window=10, min_periods=1).mean()
        score += 2.0 if obv_series.iloc[-1] > obv_ma.iloc[-1] else -2.0
        
        return float(np.clip(score, -10.0, 10.0))
    except Exception as e:
        logger.error(f"Erro no cálculo do OBV score: {e}")
        return 0.0

def check_trailing_stop(ticker: str, position: dict, current_price: float,
                        trailing_data: dict, params: dict) -> dict:
    """Verifica se o trailing stop foi ativado para um ticker."""
    result = {"ticker": ticker, "triggered": False}
    
    try:
        qty = position.get("quantity", 0)
        avg_price = position.get("avg_price", 0)
        
        if qty <= 0 or avg_price <= 0:
            return result
        
        # Inicializa dados de trailing se não existirem
        if ticker not in trailing_data:
            trailing_data[ticker] = {
                "highest_price": current_price,
                "stop_level": current_price * (1 - params.get("trailing_stop_pct", 3.0) / 100)
            }
        
        highest_price = trailing_data[ticker]["highest_price"]
        stop_pct = params.get("trailing_stop_pct", 3.0) / 100
        
        # Atualiza highest_price e stop_level se o preço subiu
        if current_price > highest_price:
            highest_price = current_price
            new_stop = current_price * (1 - stop_pct)
            trailing_data[ticker]["highest_price"] = highest_price
            trailing_data[ticker]["stop_level"] = new_stop
            logger.info(f"Atualizando trailing stop de {ticker}: novo topo ${highest_price:.2f}, stop ajustado para ${new_stop:.2f}")
        
        stop_level = trailing_data[ticker]["stop_level"]
        
        # Verifica se o trailing stop foi acionado
        if current_price <= stop_level:
            result["triggered"] = True
            result["reason"] = f"Trailing stop acionado em ${stop_level:.2f} (máx recente ${highest_price:.2f})"
        
        result["highest_price"] = highest_price
        result["stop_level"] = stop_level
        result["drawdown_pct"] = (current_price / highest_price - 1) * 100
    
    except Exception as e:
        logger.error(f"Erro no trailing stop de {ticker}: {e}")
        result["error"] = str(e)
    
    return result


# =============================================================================
# 8. PositionSizing – Cálculo do tamanho de posição
# =============================================================================
class PositionSizing:
    @staticmethod
    def calculate_VaR(df: pd.DataFrame, confidence: float = 0.95) -> float:
        """Calcula o Value at Risk (VaR) a partir dos retornos diários."""
        try:
            if df is None or df.empty or 'Daily_Return' not in df.columns:
                return 0.02
            
            returns = df['Daily_Return'].dropna()
            
            if returns.empty:
                return 0.02
            
            var = returns.quantile(1 - confidence)
            return float(abs(var))
        
        except Exception as e:
            logger.error(f"Erro ao calcular VaR: {e}")
            return 0.02

    @staticmethod
    def calculate_position_size(ticker: str, df: pd.DataFrame, account_balance: float,
                                risk_profile: str = "medium", risk_per_trade: float = 0.02,
                                daily_gap: float = 0.0, daily_goal: float = 0.0,
                                params: dict = DYNAMIC_PARAMS) -> dict:
        """Calcula o tamanho ideal de posição considerando risco e capital disponível."""
        if df is None or df.empty or 'Close' not in df.columns:
            return {"suggested_shares": 0, "risk_per_trade_pct": 0, "estimated_var_pct": 0,
                    "allocation_value": 0, "allocation_pct": 0, "commission_est": 0}
        
        price = float(df['Close'].iloc[-1])
        
        if price <= 0:
            return {"suggested_shares": 0, "risk_per_trade_pct": 0, "estimated_var_pct": 0,
                    "allocation_value": 0, "allocation_pct": 0, "commission_est": 0}
        
        # Calcular VaR e ajustar risco por perfil
        var = PositionSizing.calculate_VaR(df)
        base_risk = {"low": 0.01, "medium": 0.02, "high": 0.04, "ultra": 0.08}.get(risk_profile, risk_per_trade)
        
        # Ajustar pelo gap de recuperação
        adjustment_factor = 1.0
        if daily_goal > 0 and daily_gap > 0:
            adjustment_factor = min(1 + daily_gap / daily_goal, params.get("position_size_adjustment_max", 2.0))
        
        # Calcular risco efetivo e valor em risco
        effective_risk = base_risk * adjustment_factor
        capital_risk_amount = account_balance * effective_risk
        
        # Calcular quantidade de ações
        suggested_shares = 0
        if price > 0:
            effective_var = max(var, 0.005)  # Garantir um mínimo de VaR para evitar posições excessivas
            suggested_shares = int(capital_risk_amount / (price * effective_var))
        
        # Garantir pelo menos 1 ação se o preço permitir
        if suggested_shares < 1 and price < account_balance:
            suggested_shares = 1
        
        # Verificar se a posição cabe no saldo com comissão
        suggested_shares = PositionSizing._adjust_for_balance_and_commission(
            suggested_shares, price, account_balance
        )
        
        # Calcular valores finais
        allocation_value = suggested_shares * price
        allocation_pct = (allocation_value / account_balance * 100) if account_balance > 0 else 0.0
        commission = calculate_xp_commission(allocation_value)
        
        return {
            "suggested_shares": suggested_shares,
            "risk_per_trade_pct": effective_risk * 100,
            "estimated_var_pct": var * 100,
            "allocation_value": allocation_value,
            "allocation_pct": allocation_pct,
            "commission_est": commission
        }

    @staticmethod
    def _adjust_for_balance_and_commission(shares: int, price: float, balance: float) -> int:
        """Ajusta a quantidade de ações considerando o saldo e comissões."""
        if shares <= 0 or price <= 0 or balance <= 0:
            return 0
        
        # Reduzir gradualmente até caber no saldo
        while shares > 0:
            total_cost = shares * price + calculate_xp_commission(shares * price)
            if total_cost <= balance or shares == 1:
                break
            shares -= 1
        
        # Verificar se vale a pena (valor mínimo e comissão)
        if shares > 0:
            value = shares * price
            if value < DYNAMIC_PARAMS["min_transaction_value"]:
                return 0
        
        return shares


# =============================================================================
# 9. Funções para análise de portfólio, watchlist e plano de rebalanceamento
# =============================================================================
def load_portfolio(file_path: str) -> dict:
    """Carrega dados de portfólio a partir de um arquivo JSON."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {file_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Erro ao interpretar JSON do arquivo: {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo de portfólio: {e}")
        return {}

def analyze_portfolio(portfolio: dict, account_balance: float, risk_profile: str = "medium",
                      trailing_data: dict = None, extended_hours: bool = False, goals: dict = None) -> dict:
    """Analisa um portfólio completo, gerando recomendações para cada ativo."""
    if goals is None:
        goals = {}
    
    results = {}
    total_invested = 0.0
    total_current = 0.0
    
    if trailing_data is None:
        trailing_data = {}
    
    # Obter sentimento de mercado
    market_sent = MarketAnalysis.get_market_sentiment()
    logger.info(f"Sentimento de mercado atual: {market_sent['sentiment'].upper()} (viés {market_sent['market_bias_score']:.1f})")
    
    # Calcular dias restantes para meta de recuperação
    remaining_days = goals.get('days', 1)
    if goals.get('start_date'):
        try:
            start_date = datetime.strptime(goals['start_date'], "%Y-%m-%d")
            total_days = goals.get('days', 30)
            days_passed = (datetime.now() - start_date).days
            remaining_days = max(1, total_days - days_passed)
        except Exception as e:
            logger.warning(f"Erro ao calcular dias restantes: {e}")
    
    # Calcular meta diária de recuperação
    daily_goal = goals.get('target_recovery', 0) / max(1, remaining_days)
    
    # Obter preços em tempo real para todos os tickers
    tickers = list(portfolio.keys())
    price_map = DataLoader.get_realtime_prices_bulk(tickers)
    
    # Analisar cada ativo no portfólio
    for ticker, pos in portfolio.items():
        try:
            df = DataLoader.get_asset_data(ticker, days=60, extended_hours=extended_hours)
            
            if df.empty:
                results[ticker] = {"ticker": ticker, "error": "Dados históricos insuficientes"}
                continue
            
            # Atualizar com preço atual
            current_price = price_map.get(ticker)
            if current_price and 'Close' in df.columns:
                df.iloc[-1, df.columns.get_loc("Close")] = current_price
            
            # Obter dados da posição
            quantity = pos.get("quantity", 0)
            avg_price = pos.get("avg_price", 0)
            invested_value = quantity * avg_price
            current_value = quantity * (current_price if current_price else df['Close'].iloc[-1])
            pnl = current_value - invested_value
            
            # Calcular gap diário para a meta
            daily_gap = max(0, daily_goal - pnl)
            
            # Gerar decisão de trading
            decision = Strategy.decision_engine(
                ticker, df, pos, account_balance, risk_profile,
                DYNAMIC_PARAMS, daily_gap, daily_goal, market_sent, trailing_data,
                goals, remaining_days
            )
            
            # Calcular position sizing para compras
            if decision["decision"].upper() in ["COMPRAR", "COMPRAR PARCIAL"]:
                pos_size = PositionSizing.calculate_position_size(
                    ticker, df, account_balance, risk_profile,
                    daily_gap=daily_gap, daily_goal=daily_goal, params=DYNAMIC_PARAMS
                )
                decision["position_sizing"] = pos_size
            
            # Atualizar totais do portfólio
            if not np.isnan(current_value):
                total_invested += invested_value
                total_current += current_value
                decision["position_value"] = current_value
            
            decision["quantity"] = quantity
            results[ticker] = decision
        
        except Exception as e:
            logger.error(f"Erro ao analisar {ticker}: {e}")
            results[ticker] = {"ticker": ticker, "error": str(e)}
    
    # Criar resumo do portfólio
    portfolio_summary = {
        "total_invested": total_invested,
        "valor_atual": total_current,
        "lucro_prejuízo": total_current - total_invested,
        "lucro_prejuízo_pct": ((total_current / total_invested - 1) * 100) if total_invested > 0 else 0.0,
        "saldo_disponível": account_balance,
        "patrimônio_total": account_balance + total_current,
        "market_sentiment": market_sent,
        "meta_recuperação": goals.get('target_recovery', 0),
        "dias_restantes": remaining_days,
        "meta_diária": daily_goal
    }
    
    return {"ativos": results, "resumo": portfolio_summary}

def analyze_watchlist(watchlist: dict, account_balance: float, risk_profile: str = "medium", 
                      extended_hours: bool = False, goals: dict = None) -> dict:
    """Analisa uma watchlist, gerando recomendações para potenciais compras."""
    results = {}
    
    # Obter sentimento de mercado
    market_sent = MarketAnalysis.get_market_sentiment()
    
    # Calcular dias restantes para meta de recuperação
    remaining_days = goals.get('days', 1) if goals else 1
    if goals and goals.get('start_date'):
        try:
            start_date = datetime.strptime(goals['start_date'], "%Y-%m-%d")
            total_days = goals.get('days', 30)
            days_passed = (datetime.now() - start_date).days
            remaining_days = max(1, total_days - days_passed)
        except Exception:
            pass
    
    # Calcular meta diária de recuperação
    daily_goal = goals.get('target_recovery', 0) / max(1, remaining_days) if goals else 0
    
    # Filtrar e ordenar tickers da watchlist
    tickers = [t for t, data in watchlist.items() if data.get("monitor", False)]
    
    # Obter preços em tempo real
    price_map = DataLoader.get_realtime_prices_bulk(tickers)
    
    # Analisar cada ticker da watchlist
    for ticker, data in watchlist.items():
        if not data.get("monitor", False):
            continue
        
        try:
            df = DataLoader.get_asset_data(ticker, days=60, extended_hours=extended_hours)
            
            if df.empty:
                results[ticker] = {"ticker": ticker, "error": "Dados insuficientes"}
                continue
            
            # Atualizar último preço se disponível
            current_price = price_map.get(ticker)
            if current_price and 'Close' in df.columns:
                df.iloc[-1, df.columns.get_loc("Close")] = current_price
            
            # Criar posição vazia para análise
            fake_position = {"quantity": 0, "avg_price": 0}
            
            # Calcular gap diário para a meta
            daily_gap = daily_goal if daily_goal > 0 else 0
            
            # Gerar decisão de trading
            decision = Strategy.decision_engine(
                ticker, df, fake_position, account_balance, risk_profile,
                DYNAMIC_PARAMS, daily_gap=daily_gap, daily_goal=daily_goal, 
                market_sentiment=market_sent, goals=goals, remaining_days=remaining_days
            )
            
            # Calcular position sizing para compras
            if decision.get("decision") in ["COMPRAR", "COMPRAR PARCIAL"]:
                pos_size = PositionSizing.calculate_position_size(
                    ticker, df, account_balance, risk_profile, 
                    daily_gap=daily_gap, daily_goal=daily_goal, params=DYNAMIC_PARAMS
                )
                decision["position_sizing"] = pos_size
            
            results[ticker] = decision
        
        except Exception as e:
            logger.error(f"Erro na análise de watchlist para {ticker}: {e}")
            results[ticker] = {"ticker": ticker, "error": str(e)}
    
    return results

def generate_rebalance_plan(portfolio_analysis: dict, watchlist_analysis: dict, 
                           account_balance: float, params: dict = DYNAMIC_PARAMS) -> dict:
    """Gera um plano de rebalanceamento, considerando o portfolio atual e watchlist."""
    plan = {"sell": [], "buy": [], "rebalance": [], "stats": {}}
    
    # Inicializar contadores
    sell_capital = 0.0
    current_positions = {}
    original_value = 0.0
    
    # 1. Identificar ativos a vender
    for ticker, details in portfolio_analysis.get("ativos", {}).items():
        decision = details.get("decision", "").upper()
        qty = details.get("quantity", 0)
        price = details.get("current_price", 0)
        
        # Registrar posições atuais
        if qty > 0:
            current_positions[ticker] = {"quantity": qty, "price": price}
            original_value += qty * price
        
        # Processar decisões de venda
        if decision in ["VENDER", "REALIZAR LUCRO PARCIAL", "REDUZIR"] and qty > 0:
            sell_qty = qty if decision == "VENDER" else max(1, int(qty * 0.5))
            operation_value = sell_qty * price
            
            # Verificar valor mínimo para transação
            if operation_value < params.get("min_transaction_value", 50.0):
                logger.info(f"Ignorando venda de {ticker}: valor muito baixo (${operation_value:.2f})")
                continue
            
            commission = calculate_xp_commission(operation_value)
            capital_freed = operation_value - commission
            sell_capital += capital_freed
            
            plan["sell"].append({
                "ticker": ticker,
                "sell_quantity": sell_qty,
                "current_price": price,
                "operation_value": operation_value,
                "commission": commission,
                "capital_freed": capital_freed,
                "reason": details.get("justificativa", ""),
                "remaining_qty": qty - sell_qty,
                "score": details.get("total_score", 0)
            })
    
    # 2. Identificar candidatos de compra na watchlist
    buy_candidates = []
    for ticker, details in watchlist_analysis.items():
        decision = details.get("decision", "").upper()
        if decision in ["COMPRAR", "COMPRAR PARCIAL"]:
            score = details.get("total_score", 0)
            pos_size = details.get("position_sizing", {})
            suggested_qty = pos_size.get("suggested_shares", 0)
            price = details.get("current_price", 0)
            
            # Verificar quantidade sugerida e preço
            if suggested_qty < 1 or price <= 0:
                continue
            
            # Verificar score mínimo para rebalanceamento
            if score < params.get("rebalance_score_threshold", 40):
                continue
            
            buy_candidates.append({
                "ticker": ticker,
                "score": score,
                "suggested_qty": suggested_qty,
                "current_price": price,
                "reason": details.get("justificativa", ""),
                "operation_value": suggested_qty * price
            })
    
    # 3. Identificar possíveis rebalanceamentos dentro do portfólio
    rebalance_candidates = []
    for ticker, details in portfolio_analysis.get("ativos", {}).items():
        score = details.get("total_score", 0)
        qty = details.get("quantity", 0)
        price = details.get("current_price", 0)
        position_profit_pct = details.get("position_profit_pct", 0)
        
        # Verificar se o ativo tem baixo desempenho mas não está na lista de venda
        if (score < -20 or position_profit_pct < params.get("min_sell_performance", -3.0)) and qty > 0:
            already_in_sell = any(s['ticker'] == ticker for s in plan["sell"])
            
            if not already_in_sell:
                # Calcular quantidade a vender para rebalanceamento
                sell_qty = int(qty * 0.5)  # Vender metade para rebalancear
                if sell_qty < 1:
                    continue
                
                operation_value = sell_qty * price
                if operation_value < params.get("min_transaction_value", 50.0):
                    continue
                
                commission = calculate_xp_commission(operation_value)
                capital_freed = operation_value - commission
                
                rebalance_candidates.append({
                    "ticker": ticker,
                    "sell_quantity": sell_qty,
                    "current_price": price,
                    "operation_value": operation_value,
                    "commission": commission,
                    "capital_freed": capital_freed,
                    "score": score,
                    "profit_pct": position_profit_pct
                })
    
    # Ordenar candidatos para rebalanceamento (piores scores primeiro)
    rebalance_candidates.sort(key=lambda x: x["score"])
    
    # 4. Ordenar candidatos para compra (melhores scores primeiro)
    buy_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # 5. Verificar se é necessário rebalancear além das vendas já planejadas
    additional_capital_needed = 0
    for buy in buy_candidates[:3]:  # Considerar os top 3 candidatos
        additional_capital_needed += buy["operation_value"] + calculate_xp_commission(buy["operation_value"])
    
    available_capital = account_balance + sell_capital
    capital_gap = max(0, additional_capital_needed - available_capital)
    
    # 6. Se precisar de mais capital, considerar rebalanceamento
    if capital_gap > 0 and rebalance_candidates:
        for cand in rebalance_candidates:
            if capital_gap <= 0:
                break
                
            plan["rebalance"].append({
                "ticker": cand["ticker"],
                "sell_quantity": cand["sell_quantity"],
                "current_price": cand["current_price"],
                "operation_value": cand["operation_value"],
                "commission": cand["commission"],
                "capital_freed": cand["capital_freed"],
                "reason": f"Rebalanceamento - Score baixo ({cand['score']:.2f}) e desempenho inadequado ({cand['profit_pct']:.2f}%)"
            })
            
            sell_capital += cand["capital_freed"]
            capital_gap -= cand["capital_freed"]
            available_capital += cand["capital_freed"]
    
    # 7. Definir compras com o capital disponível
    available_capital = account_balance + sell_capital
    total_commission = 0
    
    # Ordenar novamente por score (pode ter mudado com vendas adicionais)
    buy_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    for candidate in buy_candidates:
        # Calcular quantidade máxima comprável com o capital restante
        max_affordable_qty = int((available_capital * 0.95) / candidate["current_price"]) if candidate["current_price"] > 0 else 0
        qty = min(candidate["suggested_qty"], max_affordable_qty)
        
        # Se não podemos comprar pelo menos 1, pular
        if qty < 1 or available_capital < params.get("min_transaction_value", 50.0):
            continue
        
        price = candidate["current_price"]
        operation_value = qty * price
        commission = calculate_xp_commission(operation_value)
        total_cost = operation_value + commission
        
        # Adicionar à lista de compras
        plan["buy"].append({
            "ticker": candidate["ticker"],
            "buy_quantity": qty,
            "current_price": price,
            "operation_value": operation_value,
            "commission": commission,
            "total_cost": total_cost,
            "reason": candidate["reason"]
        })
        
        # IMPORTANTE: Reduzir o capital disponível para próximas recomendações
        available_capital -= total_cost
        total_commission += commission
        
        # Limitar número de compras para não diluir demais a carteira
        if len(plan["buy"]) >= params.get("max_portfolio_assets", 5) // 2 or available_capital < 100:
            break
    
    # 8. Calcular estatísticas do plano de rebalanceamento
    initial_capital = account_balance
    capital_from_sales = sum(item["capital_freed"] for item in plan["sell"] + plan["rebalance"])
    capital_for_buys = sum(item["total_cost"] for item in plan["buy"])
    
    plan["stats"] = {
        "saldo_inicial": initial_capital,
        "capital_liberado_vendas": capital_from_sales,
        "capital_total_disponivel": initial_capital + capital_from_sales,
        "capital_usado_compras": capital_for_buys,
        "saldo_remanescente": initial_capital + capital_from_sales - capital_for_buys,
        "total_comissoes": total_commission,
        "numero_vendas": len(plan["sell"]),
        "numero_rebalanceamentos": len(plan["rebalance"]),
        "numero_compras": len(plan["buy"])
    }

    # Add debug logging before return
    try:
        logger.debug(f"Returning rebalance plan: {len(plan['sell'])} sells, {len(plan['rebalance'])} rebalances, {len(plan['buy'])} buys")
        return plan
    except Exception as e:
        logger.error(f"Error in generate_rebalance_plan when returning: {e}")
        # Return a minimal valid plan to avoid complete failure
        return {
            "sell": [],
            "buy": [],
            "rebalance": [],
            "stats": {
                "saldo_inicial": account_balance,
                "capital_liberado_vendas": 0,
                "capital_total_disponivel": account_balance,
                "capital_usado_compras": 0,
                "saldo_remanescente": account_balance,
                "total_comissoes": 0,
                "numero_vendas": 0,
                "numero_rebalanceamentos": 0,
                "numero_compras": 0
            }
        }
    
    return plan
    

def calculate_xp_commission(value: float) -> float:
    """
    Calcula a comissão da XP com base no valor da operação.
    Nova estrutura de comissões: $1.00 para operações a partir de $50.
    """
    if value < 50:
        return 0.0
    else:
        return 1.00

def create_backtest(portfolio_history: list, benchmark_ticker: str = "SPY", 
               period: int = 30, risk_free_rate: float = 0.03) -> dict:
    """
    Cria um backtest comparando o desempenho do portfólio com um benchmark.
    
    Args:
        portfolio_history: Lista de snapshots diários do portfólio
        benchmark_ticker: Ticker do benchmark (padrão: SPY)
        period: Período em dias para o backtest
        risk_free_rate: Taxa livre de risco anualizada
        
    Returns:
        Dict com métricas de desempenho
    """
    # Verificar se há dados suficientes
    if not portfolio_history or len(portfolio_history) < 2:
        return {
            "error": "Dados insuficientes para backtest",
            "metrics": {}
        }
    
    try:
        # Extrair retornos diários do portfólio
        portfolio_values = [snapshot.get("total_value", 0) for snapshot in portfolio_history]
        portfolio_returns = []
        
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i-1] > 0:
                ret = (portfolio_values[i] / portfolio_values[i-1]) - 1
                portfolio_returns.append(ret)
            else:
                portfolio_returns.append(0)
        
        # Obter dados do benchmark
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period)
        
        benchmark_data = DataLoader.get_asset_data(benchmark_ticker, days=period)
        
        if benchmark_data.empty:
            return {
                "error": f"Não foi possível obter dados para o benchmark {benchmark_ticker}",
                "portfolio_metrics": calculate_performance_metrics(portfolio_returns, risk_free_rate)
            }
        
        # Calcular retornos do benchmark
        benchmark_returns = benchmark_data['Daily_Return'].dropna().tolist()
        
        # Limitar ao mesmo comprimento
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]
        
        # Calcular métricas
        portfolio_metrics = calculate_performance_metrics(portfolio_returns, risk_free_rate)
        benchmark_metrics = calculate_performance_metrics(benchmark_returns, risk_free_rate)
        
        # Calcular correlação entre portfólio e benchmark
        correlation = pd.Series(portfolio_returns).corr(pd.Series(benchmark_returns))
        
        # Calcular Information Ratio
        active_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
        information_ratio = (sum(active_returns) / len(active_returns)) / (pd.Series(active_returns).std() or 1)
        
        return {
            "portfolio_metrics": portfolio_metrics,
            "benchmark_metrics": benchmark_metrics,
            "correlation": correlation,
            "information_ratio": information_ratio,
            "tracking_error": pd.Series(active_returns).std(),
            "active_return": sum(active_returns) / len(active_returns),
            "win_rate": sum(1 for r in active_returns if r > 0) / len(active_returns),
            "benchmark_ticker": benchmark_ticker,
            "period_days": min_length,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erro ao realizar backtest: {e}")
        return {
            "error": f"Erro ao realizar backtest: {str(e)}",
            "metrics": {}
        }

def calculate_performance_metrics(returns: list, risk_free_rate: float = 0.03) -> dict:
    """
    Calcula métricas de desempenho para uma lista de retornos.
    
    Args:
        returns: Lista de retornos diários
        risk_free_rate: Taxa livre de risco anualizada
        
    Returns:
        Dict com métricas de desempenho
    """
    if not returns:
        return {}
    
    # Converter para array numpy
    returns_array = np.array(returns)
    
    # Retorno total
    total_return = (np.prod([1 + r for r in returns]) - 1) * 100
    
    # Retorno anualizado (assumindo 252 dias de negociação por ano)
    annual_factor = 252 / len(returns)
    annual_return = ((1 + total_return / 100) ** annual_factor - 1) * 100
    
    # Volatilidade (anualizada)
    volatility = returns_array.std() * np.sqrt(252) * 100
    
    # Retorno diário médio
    avg_daily_return = returns_array.mean() * 100
    
    # Retorno livre de risco diário
    daily_risk_free = risk_free_rate / 252
    
    # Sharpe Ratio
    excess_returns = returns_array - daily_risk_free
    sharpe_ratio = excess_returns.mean() / (returns_array.std() or 0.0001) * np.sqrt(252)
    
    # Sortino Ratio (apenas downside risk)
    downside_returns = np.array([min(r - daily_risk_free, 0) for r in returns])
    downside_std = downside_returns.std() or 0.0001
    sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252)
    
    # Máximo Drawdown
    cumulative = np.cumprod(1 + returns_array)
    max_drawdown = 0
    peak = cumulative[0]
    
    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    # Dias positivos/negativos
    positive_days = sum(1 for r in returns if r > 0)
    negative_days = sum(1 for r in returns if r < 0)
    
    return {
        "total_return_pct": total_return,
        "annual_return_pct": annual_return,
        "volatility_pct": volatility,
        "avg_daily_return_pct": avg_daily_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown_pct": max_drawdown * 100,
        "positive_days": positive_days,
        "negative_days": negative_days,
        "win_rate": positive_days / len(returns) if len(returns) > 0 else 0,
        "risk_reward_ratio": abs(annual_return / volatility) if volatility > 0 else 0,
        "calmar_ratio": abs(annual_return / (max_drawdown * 100)) if max_drawdown > 0 else 0
    }

# =============================================================================
# Função auxiliar para exportar resultados para diferentes formatos
# =============================================================================
def export_results(results: dict, file_path: str, format_type: str = "json") -> bool:
    """Exporta os resultados para um arquivo no formato especificado."""
    try:
        if format_type == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
        elif format_type == "csv":
            if not results or "ativos" not in results:
                return False
            
            # Exportar portfólio
            portfolio_df = pd.DataFrame([
                {
                    "ticker": ticker,
                    "quantidade": details.get("quantity", 0),
                    "preco_atual": details.get("current_price", 0),
                    "valor_posicao": details.get("position_value", 0),
                    "lucro_prejuizo_pct": details.get("position_profit_pct", 0),
                    "lucro_prejuizo_valor": details.get("position_profit_value", 0),
                    "score_total": details.get("total_score", 0),
                    "score_tecnico": details.get("technical_score", 0),
                    "score_qualitativo": details.get("qualitative_score", 0),
                    "score_fundamental": details.get("fundamental_score", 0),
                    "decisao": details.get("decision", "AGUARDAR"),
                    "justificativa": details.get("justificativa", "")
                }
                for ticker, details in results["ativos"].items() if "error" not in details
            ])
            
            portfolio_df.to_csv(f"{file_path}_portfolio.csv", index=False)
            
            # Exportar resumo
            resumo = results.get("resumo", {})
            resumo_df = pd.DataFrame([resumo])
            resumo_df.to_csv(f"{file_path}_resumo.csv", index=False)
            
        else:
            logger.error(f"Formato de exportação não suportado: {format_type}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Erro ao exportar resultados: {e}")
        return False

# =============================================================================
# Classe para integração com APIs externas de trading
# =============================================================================
class TradingAPI:
    """Interface para integração com APIs de corretoras."""
    
    def __init__(self, api_key: str = None, api_secret: str = None, sandbox: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.authenticated = False
        self.session = requests.Session()
    
    def authenticate(self) -> bool:
        """Autentica com a API da corretora."""
        # Implementação específica para cada corretora
        logger.warning("Método authenticate() não implementado para esta API")
        return False
    
    def get_account_balance(self) -> float:
        """Obtém o saldo disponível na conta."""
        # Implementação específica para cada corretora
        logger.warning("Método get_account_balance() não implementado para esta API")
        return 0.0
    
    def get_positions(self) -> dict:
        """Obtém as posições atuais na carteira."""
        # Implementação específica para cada corretora
        logger.warning("Método get_positions() não implementado para esta API")
        return {}
    
    def place_order(self, ticker: str, qty: int, side: str, order_type: str = "market") -> dict:
        """Envia uma ordem de compra ou venda."""
        # Implementação específica para cada corretora
        logger.warning("Método place_order() não implementado para esta API")
        return {"success": False, "message": "Método não implementado"}
    
    def get_order_status(self, order_id: str) -> dict:
        """Verifica o status de uma ordem."""
        # Implementação específica para cada corretora
        logger.warning("Método get_order_status() não implementado para esta API")
        return {"status": "unknown"}

# =============================================================================
# Classe para notificações
# =============================================================================
class NotificationSystem:
    """Sistema de notificações para alertas e atualizações."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
    
    def send_notification(self, title: str, message: str, level: str = "info") -> bool:
        """Envia uma notificação genérica."""
        if not self.enabled:
            return False
        
        try:
            # Log da notificação
            log_method = getattr(logger, level.lower(), logger.info)
            log_method(f"Notificação: {title} - {message}")
            
            # Verificar métodos habilitados
            methods = self.config.get("methods", {})
            
            success = False
            
            # Email
            if methods.get("email", {}).get("enabled", False):
                email_success = self._send_email(title, message, level)
                success = success or email_success
            
            # SMS
            if methods.get("sms", {}).get("enabled", False):
                sms_success = self._send_sms(title, message)
                success = success or sms_success
            
            # Webhook (por exemplo, para Discord, Slack, etc.)
            if methods.get("webhook", {}).get("enabled", False):
                webhook_success = self._send_webhook(title, message, level)
                success = success or webhook_success
            
            return success
        
        except Exception as e:
            logger.error(f"Erro ao enviar notificação: {e}")
            return False
    
    def _send_email(self, title: str, message: str, level: str = "info") -> bool:
        """Envia notificação por email."""
        try:
            email_config = self.config.get("methods", {}).get("email", {})
            
            if not email_config:
                return False
            
            # Aqui seria implementado o envio de email
            logger.info(f"Simulando envio de email: {title} - {message}")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao enviar email: {e}")
            return False
    
    def _send_sms(self, title: str, message: str) -> bool:
        """Envia notificação por SMS."""
        try:
            sms_config = self.config.get("methods", {}).get("sms", {})
            
            if not sms_config:
                return False
            
            # Aqui seria implementado o envio de SMS
            logger.info(f"Simulando envio de SMS: {title} - {message}")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao enviar SMS: {e}")
            return False
    
    def _send_webhook(self, title: str, message: str, level: str = "info") -> bool:
        """Envia notificação via webhook (Discord, Slack, etc)."""
        try:
            webhook_config = self.config.get("methods", {}).get("webhook", {})
            
            if not webhook_config:
                return False
            
            url = webhook_config.get("url", "")
            if not url:
                return False
            
            # Formatar payload
            payload = {
                "content": message,
                "username": webhook_config.get("username", "Trading System"),
                "embeds": [{
                    "title": title,
                    "description": message,
                    "color": {
                        "info": 3447003,  # Azul
                        "warning": 16776960,  # Amarelo
                        "error": 15158332,  # Vermelho
                        "success": 3066993  # Verde
                    }.get(level, 10181046)  # Cinza como padrão
                }]
            }
            
            # Enviar para webhook
            response = http_session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            return True
        except Exception as e:
            logger.error(f"Erro ao enviar webhook: {e}")
            return False
    
    def alert_trade_recommendation(self, ticker: str, action: str, 
                                  quantity: int, price: float, reason: str) -> bool:
        """Envia alerta de recomendação de trade."""
        title = f"Recomendação de {action}: {ticker}"
        message = (
            f"Recomendação para {action} {quantity} ação(ões) de {ticker}\n"
            f"Preço: ${price:.2f}\n"
            f"Valor Total: ${quantity * price:.2f}\n"
            f"Razão: {reason}"
        )
        
        return self.send_notification(title, message, "info")
    
    def alert_execution(self, trade_data: dict) -> bool:
        """Envia alerta de execução de trade."""
        action = trade_data.get("action", "")
        ticker = trade_data.get("ticker", "")
        
        title = f"{action} Executado: {ticker}"
        message = (
            f"{action} de {trade_data.get('quantity', 0)} {ticker} executado\n"
            f"Preço: ${trade_data.get('price', 0):.2f}\n"
            f"Valor: ${trade_data.get('gross_value', 0):.2f}\n"
            f"Comissão: ${trade_data.get('commission', 0):.2f}"
        )
        
        return self.send_notification(title, message, "success")

# =============================================================================
# Ajustes finais e configurações adicionais
# =============================================================================

def load_configuration(config_file: str) -> dict:
    """Carrega configurações a partir de um arquivo JSON."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Erro ao carregar configuração de {config_file}: {e}")
        return {}

def setup_logging(config: dict = None) -> None:
    """Configura o sistema de logging com base nas configurações."""
    if not config:
        return
    
    log_config = config.get("logging", {})
    
    if not log_config.get("enabled", True):
        logger.disabled = True
        return
    
    log_level = log_config.get("level", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    log_file = log_config.get("file", "trading_system.log")
    
    if log_config.get("rotate", False):
        from logging.handlers import RotatingFileHandler
        max_size = log_config.get("max_size", 1024 * 1024)  # 1MB padrão
        backup_count = log_config.get("backup_count", 5)
        
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_size, backupCount=backup_count
        )
    else:
        file_handler = logging.FileHandler(log_file)
    
    formatter = logging.Formatter(
        log_config.get("format", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    file_handler.setFormatter(formatter)
    
    # Remover handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Adicionar novos handlers
    logger.addHandler(file_handler)
    
    if log_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

# =============================================================================
# Função para verificar atualizações do sistema
# =============================================================================
def check_for_updates(current_version: str = "2.0.0") -> dict:
    """Verifica se há atualizações disponíveis para o sistema de trading."""
    try:
        # Esta função é um placeholder e normalmente faria uma chamada a uma API
        # para verificar atualizações disponíveis.
        logger.info(f"Verificando atualizações. Versão atual: {current_version}")
        
        # Aqui seria feita uma chamada real para verificar atualizações
        # Por enquanto, retornamos um resultado simulado
        return {
            "current_version": current_version,
            "latest_version": "2.0.0",
            "update_available": False,
            "update_url": "",
            "release_notes": ""
        }
    except Exception as e:
        logger.error(f"Erro ao verificar atualizações: {e}")
        return {
            "error": str(e),
            "current_version": current_version,
            "update_available": False
        }

# =============================================================================
# Funções para diagnóstico e solução de problemas
# =============================================================================
def run_diagnostics() -> dict:
    """Executa diagnósticos no sistema e ambiente de execução."""
    results = {
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "time": datetime.now().isoformat()
        },
        "dependencies": {},
        "connectivity": {},
        "data_access": {}
    }
    
    # Verificar dependências
    try:
        results["dependencies"]["numpy"] = np.__version__
    except Exception as e:
        results["dependencies"]["numpy"] = f"Erro: {str(e)}"
    
    try:
        results["dependencies"]["pandas"] = pd.__version__
    except Exception as e:
        results["dependencies"]["pandas"] = f"Erro: {str(e)}"
    
    try:
        results["dependencies"]["yfinance"] = yf.__version__
    except Exception as e:
        results["dependencies"]["yfinance"] = f"Erro: {str(e)}"
    
    try:
        results["dependencies"]["sklearn"] = sklearn.__version__
    except Exception as e:
        results["dependencies"]["sklearn"] = f"Erro: {str(e)}"
    
    try:
        results["dependencies"]["requests"] = requests.__version__
    except Exception as e:
        results["dependencies"]["requests"] = f"Erro: {str(e)}"
    
    # Verificar conectividade
    try:
        response = requests.get("https://query1.finance.yahoo.com", timeout=5)
        results["connectivity"]["yahoo_finance"] = {
            "status": response.status_code,
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        results["connectivity"]["yahoo_finance"] = {"error": str(e)}
    
    # Testar acesso a dados
    try:
        test_ticker = "SPY"
        df = DataLoader.get_asset_data(test_ticker, days=5)
        results["data_access"]["yfinance"] = {
            "success": not df.empty,
            "rows": len(df) if not df.empty else 0,
            "columns": list(df.columns) if not df.empty else []
        }
    except Exception as e:
        results["data_access"]["yfinance"] = {"error": str(e)}
    
    return results

def verify_portfolio_data(portfolio: dict) -> dict:
    """Verifica a integridade dos dados de portfólio."""
    validation = {
        "status": "success",
        "errors": [],
        "warnings": []
    }
    
    if not portfolio:
        validation["status"] = "error"
        validation["errors"].append("Portfólio vazio")
        return validation
    
    # Verificar cada posição no portfólio
    for ticker, position in portfolio.items():
        # Verificar se é um ticker válido
        if not isinstance(ticker, str) or len(ticker) < 1:
            validation["errors"].append(f"Ticker inválido: {ticker}")
            validation["status"] = "error"
        
        # Verificar quantidade
        qty = position.get("quantity", 0)
        if not isinstance(qty, (int, float)) or qty <= 0:
            validation["errors"].append(f"Quantidade inválida para {ticker}: {qty}")
            validation["status"] = "error"
        
        # Verificar preço médio
        avg_price = position.get("avg_price", 0)
        if not isinstance(avg_price, (int, float)) or avg_price <= 0:
            validation["errors"].append(f"Preço médio inválido para {ticker}: {avg_price}")
            validation["status"] = "error"
        
        # Verificar última compra/venda se existir
        if "last_buy" in position:
            last_buy = position["last_buy"]
            if not isinstance(last_buy, dict) or "price" not in last_buy:
                validation["warnings"].append(f"Dados de última compra inválidos para {ticker}")
        
        if "last_sell" in position:
            last_sell = position["last_sell"]
            if not isinstance(last_sell, dict) or "price" not in last_sell:
                validation["warnings"].append(f"Dados de última venda inválidos para {ticker}")
    
    return validation

# =============================================================================
# Interface de linha de comando auxiliar
# =============================================================================
def parse_command_line_args():
    """Parse argumentos de linha de comando específicos para o módulo."""
    parser = argparse.ArgumentParser(description="Sistema de Trading Swing Trade Agressivo")
    
    # Argumentos básicos
    parser.add_argument('--json-file', type=str, help='Arquivo JSON com portfolio, watchlist, account_balance e goals')
    parser.add_argument('--risk', type=str, default="high", choices=['low','medium','high','ultra'], help='Perfil de risco (default: high)')
    parser.add_argument('--output-format', type=str, default="text", choices=['text','json','table'], help='Formato de saída')
    parser.add_argument('--use-polygon', action='store_true', help='Usar Polygon.io para dados (se disponível)')
    parser.add_argument('--extended', action='store_true', help='Incluir dados de extended hours (pré e pós mercado)')
    
    # Argumentos adicionais
    parser.add_argument('--config', type=str, help='Arquivo de configuração adicional')
    parser.add_argument('--validate-tickers', action='store_true', help='Validar tickers antes de analisar (mais lento)')
    parser.add_argument('--debug', action='store_true', help='Exibir informações de debug durante execução')
    parser.add_argument('--export-file', type=str, help='Exportar resultados para arquivo')
    parser.add_argument('--export-format', type=str, default="json", choices=['json','csv'], help='Formato de exportação')
    parser.add_argument('--simulate', action='store_true', help='Simular execução de rebalanceamento sem realizar ordens')
    parser.add_argument('--diagnostics', action='store_true', help='Executar diagnósticos do sistema')
    parser.add_argument('--version', action='store_true', help='Exibir versão do sistema')
    parser.add_argument('--backtest', action='store_true', help='Executar backtest com dados históricos')
    parser.add_argument('--backtest-days', type=int, default=30, help='Período para backtest em dias')
    parser.add_argument('--optimize', action='store_true', help='Executar otimização de carteira')
    parser.add_argument('--interactive', action='store_true', help='Iniciar modo interativo')
    
    return parser.parse_args()

# =============================================================================
# Modo Interativo
# =============================================================================
def interactive_mode():
    """Inicia o sistema em modo interativo."""
    print("\n" + "="*50)
    print("Sistema de Trading Swing Trade Agressivo - Modo Interativo")
    print("="*50)
    
    # Carregar dados
    portfolio_file = input("\nInforme o caminho do arquivo de portfólio (JSON): ")
    
    try:
        data = load_portfolio(portfolio_file)
        if not data:
            print("Erro: Arquivo inválido ou vazio.")
            return
    except Exception as e:
        print(f"Erro ao carregar arquivo: {e}")
        return
    
    portfolio = data.get("portfolio", {})
    watchlist = data.get("watchlist", {})
    account_balance = data.get("account_balance", 0.0)
    goals = data.get("goals", {})
    
    print(f"\nPortfólio carregado: {len(portfolio)} ativos")
    print(f"Watchlist: {len(watchlist)} ativos")
    print(f"Saldo: ${account_balance:,.2f}")
    
    if goals:
        print(f"Meta de recuperação: ${goals.get('target_recovery', 0):,.2f} em {goals.get('days', 0)} dias")
    
    # Menu de opções
    while True:
        print("\n" + "-"*50)
        print("Escolha uma opção:")
        print("1. Analisar carteira")
        print("2. Analisar watchlist")
        print("3. Gerar plano de rebalanceamento")
        print("4. Otimizar carteira")
        print("5. Simular execução de rebalanceamento")
        print("6. Executar backtest")
        print("7. Diagnosticar sistema")
        print("8. Verificar atualizações")
        print("9. Exportar dados")
        print("0. Sair")
        
        choice = input("\nOpção: ")
        
        if choice == "1":
            risk_profile = input("Perfil de risco (low, medium, high, ultra) [high]: ") or "high"
            trailing_data = {}
            
            print("\nAnalisando carteira...")
            portfolio_analysis = analyze_portfolio(
                portfolio, account_balance, risk_profile,
                trailing_data, extended_hours=False, goals=goals
            )
            
            output_format = input("Formato de saída (text, table, json) [text]: ") or "text"
            
            if output_format == "json":
                print(json.dumps(portfolio_analysis, indent=2, default=str))
            elif output_format == "table" and HAS_TABULATE:
                output_table_results(portfolio_analysis, {}, {"sell": [], "buy": [], "rebalance": [], "stats": {}})
            else:
                output_text_results(portfolio_analysis, {}, {"sell": [], "buy": [], "rebalance": [], "stats": {}})
        
        elif choice == "2":
            risk_profile = input("Perfil de risco (low, medium, high, ultra) [high]: ") or "high"
            
            print("\nAnalisando watchlist...")
            watchlist_analysis = analyze_watchlist(
                watchlist, account_balance, risk_profile,
                extended_hours=False, goals=goals
            )
            
            output_format = input("Formato de saída (text, table, json) [text]: ") or "text"
            
            if output_format == "json":
                print(json.dumps(watchlist_analysis, indent=2, default=str))
            elif output_format == "table" and HAS_TABULATE:
                # Criar uma tabela específica para watchlist
                watchlist_table = []
                for ticker, details in watchlist_analysis.items():
                    if "error" in details:
                        continue
                    
                    watchlist_table.append([
                        ticker,
                        f"${details.get('current_price', 0):,.2f}",
                        f"{details.get('total_score', 0):.2f}",
                        details.get('decision', "AGUARDAR"),
                        details.get('justificativa', "")
                    ])
                
                if watchlist_table:
                    headers = ["Ticker", "Preço", "Score", "Decisão", "Justificativa"]
                    print(tabulate(watchlist_table, headers=headers, tablefmt="grid"))
                else:
                    print("Nenhum resultado na watchlist.")
            else:
                # Saída simplificada para watchlist
                for ticker, details in watchlist_analysis.items():
                    if "error" in details:
                        continue
                    
                    print(f"\n** {ticker}: **")
                    print(f"Preço: ${details.get('current_price', 0):,.2f}")
                    print(f"Score: {details.get('total_score', 0):.2f}")
                    print(f"Decisão: {details.get('decision', 'AGUARDAR')}")
                    print(f"Justificativa: {details.get('justificativa', '')}")
        
        elif choice == "3":
            risk_profile = input("Perfil de risco (low, medium, high, ultra) [high]: ") or "high"
            trailing_data = {}
            
            print("\nAnalisando carteira...")
            portfolio_analysis = analyze_portfolio(
                portfolio, account_balance, risk_profile,
                trailing_data, extended_hours=False, goals=goals
            )
            
            print("Analisando watchlist...")
            watchlist_analysis = analyze_watchlist(
                watchlist, account_balance, risk_profile,
                extended_hours=False, goals=goals
            )
            
            print("Gerando plano de rebalanceamento...")
            rebalance_plan = generate_rebalance_plan(
                portfolio_analysis, watchlist_analysis,
                account_balance, DYNAMIC_PARAMS
            )
            
            output_format = input("Formato de saída (text, table, json) [text]: ") or "text"
            
            if output_format == "json":
                print(json.dumps(rebalance_plan, indent=2, default=str))
            elif output_format == "table" and HAS_TABULATE:
                output_table_results(portfolio_analysis, watchlist_analysis, rebalance_plan)
            else:
                output_text_results(portfolio_analysis, watchlist_analysis, rebalance_plan)
        
        elif choice == "4":
            risk_profile = input("Perfil de risco (low, medium, high, ultra) [high]: ") or "high"
            max_positions = int(input("Número máximo de posições [15]: ") or "15")
            target_cash_reserve_pct = float(input("Reserva de caixa % [10]: ") or "10") / 100
            
            print("\nOtimizando carteira...")
            
            market_sentiment = MarketAnalysis.get_market_sentiment()
            optimizer = PortfolioOptimizer(
                portfolio, watchlist, account_balance, risk_profile, market_sentiment
            )
            
            results = optimizer.optimize(
                max_positions=max_positions,
                target_cash_reserve_pct=target_cash_reserve_pct
            )
            
            print("\n===== RESULTADO DA OTIMIZAÇÃO =====")
            print(f"Valor total da carteira: ${results['total_portfolio_value']:,.2f}")
            print(f"Saldo atual: ${results['current_cash']:,.2f}")
            print(f"Reserva de caixa alvo: ${results['target_cash_reserve']:,.2f}")
            
            print("\nAtivos a manter:")
            for asset in results['assets_to_hold']:
                print(f"- {asset['ticker']}: {asset['qty']} ações, ${asset['current_value']:,.2f}")
            
            print("\nAtivos a vender:")
            for asset in results['assets_to_sell']:
                print(f"- {asset['ticker']}: {asset['sell_qty']} ações, ${asset['price']:,.2f}/ação")
            
            print("\nAtivos a comprar:")
            for asset in results['assets_to_buy']:
                print(f"- {asset['ticker']}: {asset['buy_qty']} ações, ${asset['price']:,.2f}/ação")
        
        elif choice == "5":
            # Simulação de execução
            print("\nSimulando execução de rebalanceamento...")
            
            # Primeiro gerar o plano
            risk_profile = input("Perfil de risco (low, medium, high, ultra) [high]: ") or "high"
            trailing_data = {}
            
            portfolio_analysis = analyze_portfolio(
                portfolio, account_balance, risk_profile,
                trailing_data, extended_hours=False, goals=goals
            )
            
            watchlist_analysis = analyze_watchlist(
                watchlist, account_balance, risk_profile,
                extended_hours=False, goals=goals
            )
            
            rebalance_plan = generate_rebalance_plan(
                portfolio_analysis, watchlist_analysis,
                account_balance, DYNAMIC_PARAMS
            )
            
            # Criar executor
            executor = PortfolioExecutor(portfolio, account_balance)
            
            # Simulação
            print("\nSimulando execução...")
            results = executor.execute_rebalance_plan(rebalance_plan)
            
            print("\n===== RESULTADO DA SIMULAÇÃO =====")
            print(f"Saldo inicial: ${results['starting_balance']:,.2f}")
            print(f"Capital liberado: ${results['capital_freed']:,.2f}")
            print(f"Capital investido: ${results['capital_invested']:,.2f}")
            print(f"Saldo final: ${results['ending_balance']:,.2f}")
            print(f"Comissões totais: ${results['total_commission']:,.2f}")
            
            print(f"\nOperações executadas: {results['executed_transactions']}")
            print(f"- Vendas: {len(results['sells_executed'])}")
            print(f"- Compras: {len(results['buys_executed'])}")
            
            if results['errors']:
                print(f"\nErros: {len(results['errors'])}")
                for error in results['errors']:
                    print(f"- {error.get('ticker', 'Unknown')}: {error.get('error', 'Unknown error')}")
            
            # Salvar resultado?
            if input("\nDeseja salvar o resultado da simulação? (s/n) [n]: ").lower() == 's':
                filename = input("Nome do arquivo: ")
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Arquivo salvo: {filename}")
        
        elif choice == "6":
            # Backtest
            print("\nExecutando backtest...")
            
            days = int(input("Período em dias [30]: ") or "30")
            benchmark = input("Ticker de referência (benchmark) [SPY]: ") or "SPY"
            
            # Simular histórico de portfólio
            # Na prática, seria usado um histórico real
            portfolio_history = []
            
            # Criar backtest
            backtest_results = create_backtest(portfolio_history, benchmark, days)
            
            if "error" in backtest_results:
                print(f"Erro no backtest: {backtest_results['error']}")
            else:
                print("\n===== RESULTADOS DO BACKTEST =====")
                
                portfolio_metrics = backtest_results.get("portfolio_metrics", {})
                benchmark_metrics = backtest_results.get("benchmark_metrics", {})
                
                print(f"Período: {backtest_results.get('period_days', 0)} dias")
                print(f"Benchmark: {backtest_results.get('benchmark_ticker', '')}")
                
                print(f"\nRetorno total da carteira: {portfolio_metrics.get('total_return_pct', 0):.2f}%")
                print(f"Retorno anualizado: {portfolio_metrics.get('annual_return_pct', 0):.2f}%")
                print(f"Volatilidade: {portfolio_metrics.get('volatility_pct', 0):.2f}%")
                print(f"Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")
                print(f"Máximo Drawdown: {portfolio_metrics.get('max_drawdown_pct', 0):.2f}%")
                
                print(f"\nRetorno total do benchmark: {benchmark_metrics.get('total_return_pct', 0):.2f}%")
                
                print(f"\nCorrelação com benchmark: {backtest_results.get('correlation', 0):.2f}")
                print(f"Information Ratio: {backtest_results.get('information_ratio', 0):.2f}")
                print(f"Retorno ativo: {backtest_results.get('active_return', 0) * 100:.2f}%")
                print(f"Tracking Error: {backtest_results.get('tracking_error', 0) * 100:.2f}%")
        
        elif choice == "7":
            # Diagnósticos
            print("\nExecutando diagnósticos do sistema...")
            
            diagnostics = run_diagnostics()
            
            print("\n===== DIAGNÓSTICOS DO SISTEMA =====")
            print(f"Python: {diagnostics['system']['python_version']}")
            print(f"Plataforma: {diagnostics['system']['platform']}")
            
            print("\nDependências:")
            for lib, version in diagnostics['dependencies'].items():
                print(f"- {lib}: {version}")
            
            print("\nConectividade:")
            for service, info in diagnostics['connectivity'].items():
                if "error" in info:
                    print(f"- {service}: Erro ({info['error']})")
                else:
                    print(f"- {service}: OK (Status {info.get('status', 'N/A')})")
            
            print("\nAcesso a dados:")
            for source, info in diagnostics['data_access'].items():
                if "error" in info:
                    print(f"- {source}: Erro ({info['error']})")
                else:
                    print(f"- {source}: {'OK' if info.get('success', False) else 'Falha'}")
        
        elif choice == "8":
            # Verificar atualizações
            print("\nVerificando atualizações...")
            
            update_info = check_for_updates("2.0.0")
            
            if "error" in update_info:
                print(f"Erro ao verificar atualizações: {update_info['error']}")
            else:
                print(f"Versão atual: {update_info['current_version']}")
                print(f"Última versão: {update_info['latest_version']}")
                
                if update_info['update_available']:
                    print("Atualização disponível!")
                    print(f"URL de download: {update_info['update_url']}")
                    
                    if update_info['release_notes']:
                        print("\nNotas da versão:")
                        print(update_info['release_notes'])
                else:
                    print("Você está usando a versão mais recente!")
        
        elif choice == "9":
            # Exportar dados
            export_format = input("Formato de exportação (json, csv) [json]: ") or "json"
            export_file = input("Nome do arquivo: ")
            
            if not export_file:
                print("Nome de arquivo inválido.")
                continue
            
            # Primeiro precisamos gerar os dados
            risk_profile = input("Perfil de risco (low, medium, high, ultra) [high]: ") or "high"
            trailing_data = {}
            
            print("\nAnalisando carteira...")
            portfolio_analysis = analyze_portfolio(
                portfolio, account_balance, risk_profile,
                trailing_data, extended_hours=False, goals=goals
            )
            
            print("Exportando dados...")
            if export_results(portfolio_analysis, export_file, export_format):
                print(f"Dados exportados para: {export_file}")
            else:
                print("Erro ao exportar dados.")
        
        elif choice == "0":
            print("\nSaindo do sistema...")
            break
        
        else:
            print("\nOpção inválida. Tente novamente.")

class RecoveryTracker:
    """Acompanha o progresso em direção à meta de recuperação."""
    
    def __init__(self, goals: dict, initial_portfolio_value: float, initial_balance: float):
        self.target_recovery = goals.get('target_recovery', 0)
        self.total_days = goals.get('days', 30)
        self.start_date = goals.get('start_date', datetime.now().strftime("%Y-%m-%d"))
        self.start_value = initial_portfolio_value + initial_balance
        self.daily_snapshots = []
        self.cumulative_pnl = 0
        
        try:
            self.start_date_obj = datetime.strptime(self.start_date, "%Y-%m-%d")
        except:
            self.start_date_obj = datetime.now()
    
    def update(self, current_portfolio_value: float, current_balance: float):
        """Atualiza o rastreador com novos valores."""
        current_value = current_portfolio_value + current_balance
        current_pnl = current_value - self.start_value
        self.cumulative_pnl = current_pnl
        
        days_passed = (datetime.now() - self.start_date_obj).days
        days_passed = max(0, min(days_passed, self.total_days))
        
        self.daily_snapshots.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "day_number": days_passed,
            "portfolio_value": current_portfolio_value,
            "cash_balance": current_balance,
            "total_value": current_value,
            "daily_pnl": current_pnl - (self.daily_snapshots[-1]["total_value"] - self.start_value if self.daily_snapshots else 0),
            "cumulative_pnl": current_pnl,
            "progress_pct": (current_pnl / self.target_recovery * 100) if self.target_recovery > 0 else 0
        })
    
    def get_status(self) -> dict:
        """Retorna o status atual do progresso de recuperação."""
        days_passed = (datetime.now() - self.start_date_obj).days
        days_passed = max(0, min(days_passed, self.total_days))
        days_remaining = max(0, self.total_days - days_passed)
        
        if not self.daily_snapshots:
            return {
                "days_passed": days_passed,
                "days_remaining": days_remaining,
                "cumulative_pnl": 0,
                "progress_pct": 0,
                "daily_required": self.target_recovery / max(1, days_remaining),
                "on_track": False,
                "projection": 0
            }
        
        last_snapshot = self.daily_snapshots[-1]
        
        # Projeção simples baseada no progresso atual
        if days_passed > 0:
            daily_avg = last_snapshot["cumulative_pnl"] / days_passed
            projection = last_snapshot["cumulative_pnl"] + (daily_avg * days_remaining)
        else:
            projection = 0
        
        # Verificar se está no caminho certo
        expected_progress = (days_passed / self.total_days) * self.target_recovery if self.total_days > 0 else 0
        on_track = last_snapshot["cumulative_pnl"] >= expected_progress
        
        return {
            "days_passed": days_passed,
            "days_remaining": days_remaining,
            "cumulative_pnl": last_snapshot["cumulative_pnl"],
            "progress_pct": last_snapshot["progress_pct"],
            "daily_required": self.target_recovery / max(1, days_remaining) if self.target_recovery > 0 else 0,
            "on_track": on_track,
            "projection": projection,
            "projected_completion_pct": (projection / self.target_recovery * 100) if self.target_recovery > 0 else 0
        }
        
    def save_to_file(self, file_path: str):
        """Salva o histórico de acompanhamento em um arquivo."""
        data = {
            "target_recovery": self.target_recovery,
            "total_days": self.total_days,
            "start_date": self.start_date,
            "start_value": self.start_value,
            "daily_snapshots": self.daily_snapshots,
            "status": self.get_status()
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar rastreamento: {e}")
            return False


class PortfolioExecutor:
    """Implementa as ações recomendadas no plano de rebalanceamento."""
    
    def __init__(self, portfolio: dict, account_balance: float):
        self.portfolio = portfolio
        self.account_balance = account_balance
        self.transaction_log = []
    
    def execute_rebalance_plan(self, rebalance_plan: dict) -> dict:
        """Executa um plano de rebalanceamento e retorna o resultado."""
        results = {
            "sells_executed": [],
            "buys_executed": [],
            "rebalances_executed": [],
            "errors": [],
            "starting_balance": self.account_balance,
            "ending_balance": self.account_balance,
            "capital_freed": 0,
            "capital_invested": 0,
            "total_commission": 0
        }
        
        # Executar vendas
        for sell in rebalance_plan.get("sell", []):
            try:
                result = self._execute_sell(
                    sell["ticker"], 
                    sell["sell_quantity"], 
                    sell["current_price"]
                )
                
                if result["success"]:
                    results["capital_freed"] += result["net_value"]
                    results["total_commission"] += result["commission"]
                    results["sells_executed"].append(result)
                else:
                    results["errors"].append(result)
            except Exception as e:
                logger.error(f"Erro ao executar venda de {sell['ticker']}: {e}")
                results["errors"].append({
                    "ticker": sell["ticker"],
                    "action": "SELL",
                    "success": False,
                    "error": str(e)
                })
        
        # Executar rebalanceamentos
        for rebalance in rebalance_plan.get("rebalance", []):
            try:
                result = self._execute_sell(
                    rebalance["ticker"], 
                    rebalance["sell_quantity"], 
                    rebalance["current_price"],
                    is_rebalance=True
                )
                
                if result["success"]:
                    results["capital_freed"] += result["net_value"]
                    results["total_commission"] += result["commission"]
                    results["rebalances_executed"].append(result)
                else:
                    results["errors"].append(result)
            except Exception as e:
                logger.error(f"Erro ao executar rebalanceamento de {rebalance['ticker']}: {e}")
                results["errors"].append({
                    "ticker": rebalance["ticker"],
                    "action": "REBALANCE",
                    "success": False,
                    "error": str(e)
                })
        
        # Atualizar saldo após vendas
        self.account_balance += results["capital_freed"]
        results["balance_after_sells"] = self.account_balance
        
        # Executar compras
        for buy in rebalance_plan.get("buy", []):
            try:
                # Verificar se ainda há saldo suficiente
                total_cost = buy["operation_value"] + buy["commission"]
                
                if total_cost > self.account_balance:
                    # Ajustar quantidade para o saldo disponível
                    adjusted_qty = self._calculate_max_affordable_quantity(
                        buy["current_price"],
                        self.account_balance
                    )
                    
                    if adjusted_qty < 1:
                        results["errors"].append({
                            "ticker": buy["ticker"],
                            "action": "BUY",
                            "success": False,
                            "error": "Saldo insuficiente para comprar qualquer quantidade"
                        })
                        continue
                    
                    logger.warning(f"Ajustando quantidade de compra para {buy['ticker']} de {buy['buy_quantity']} para {adjusted_qty} devido ao saldo disponível")
                    buy["buy_quantity"] = adjusted_qty
                    buy["operation_value"] = adjusted_qty * buy["current_price"]
                    buy["commission"] = calculate_xp_commission(buy["operation_value"])
                    buy["total_cost"] = buy["operation_value"] + buy["commission"]
                
                result = self._execute_buy(
                    buy["ticker"],
                    buy["buy_quantity"],
                    buy["current_price"]
                )
                
                if result["success"]:
                    results["capital_invested"] += result["total_cost"]
                    results["total_commission"] += result["commission"]
                    results["buys_executed"].append(result)
                else:
                    results["errors"].append(result)
            except Exception as e:
                logger.error(f"Erro ao executar compra de {buy['ticker']}: {e}")
                results["errors"].append({
                    "ticker": buy["ticker"],
                    "action": "BUY",
                    "success": False,
                    "error": str(e)
                })
        
        # Resultados finais
        results["ending_balance"] = self.account_balance
        results["executed_transactions"] = len(results["sells_executed"]) + len(results["rebalances_executed"]) + len(results["buys_executed"])
        results["timestamp"] = datetime.now().isoformat()
        
        # Adicionar resumo do portfólio atualizado
        results["updated_portfolio"] = self._calculate_portfolio_summary()
        
        return results
    
    def _execute_sell(self, ticker: str, quantity: int, price: float, is_rebalance: bool = False) -> dict:
        """Executa uma venda ou rebalanceamento."""
        if ticker not in self.portfolio:
            return {
                "ticker": ticker,
                "action": "REBALANCE" if is_rebalance else "SELL",
                "success": False,
                "error": "Ticker não encontrado no portfólio"
            }
        
        current_position = self.portfolio[ticker]
        current_qty = current_position.get("quantity", 0)
        
        if current_qty < quantity:
            return {
                "ticker": ticker,
                "action": "REBALANCE" if is_rebalance else "SELL",
                "success": False,
                "error": f"Quantidade insuficiente. Disponível: {current_qty}, Solicitado: {quantity}"
            }
        
        # Calcular valores
        gross_value = quantity * price
        commission = calculate_xp_commission(gross_value)
        net_value = gross_value - commission
        
        # Executar venda
        new_qty = current_qty - quantity
        avg_price = current_position.get("avg_price", 0)
        
        # Registrar transação
        transaction = {
            "ticker": ticker,
            "action": "REBALANCE" if is_rebalance else "SELL",
            "quantity": quantity,
            "price": price,
            "gross_value": gross_value,
            "commission": commission,
            "net_value": net_value,
            "timestamp": datetime.now().isoformat()
        }
        
        self.transaction_log.append(transaction)
        
        # Atualizar portfólio
        if new_qty > 0:
            self.portfolio[ticker]["quantity"] = new_qty
            self.portfolio[ticker]["last_sell"] = {
                "price": price,
                "quantity": quantity,
                "date": datetime.now().isoformat()
            }
        else:
            # Remover posição se quantidade = 0
            del self.portfolio[ticker]
        
        # Atualizar saldo
        self.account_balance += net_value
        
        return {
            "ticker": ticker,
            "action": "REBALANCE" if is_rebalance else "SELL",
            "success": True,
            "quantity": quantity,
            "price": price,
            "gross_value": gross_value,
            "commission": commission,
            "net_value": net_value,
            "remaining_quantity": new_qty,
            "timestamp": datetime.now().isoformat()
        }
    
    def _execute_buy(self, ticker: str, quantity: int, price: float) -> dict:
        """Executa uma compra."""
        # Calcular valores
        gross_value = quantity * price
        commission = calculate_xp_commission(gross_value)
        total_cost = gross_value + commission
        
        # Verificar saldo
        if total_cost > self.account_balance:
            return {
                "ticker": ticker,
                "action": "BUY",
                "success": False,
                "error": f"Saldo insuficiente. Disponível: ${self.account_balance:.2f}, Necessário: ${total_cost:.2f}"
            }
        
        # Verificar se já existe posição
        if ticker in self.portfolio:
            current_position = self.portfolio[ticker]
            current_qty = current_position.get("quantity", 0)
            current_avg_price = current_position.get("avg_price", 0)
            
            # Calcular novo preço médio e quantidade
            new_qty = current_qty + quantity
            new_avg_price = ((current_qty * current_avg_price) + gross_value) / new_qty
            
            # Atualizar posição
            self.portfolio[ticker]["quantity"] = new_qty
            self.portfolio[ticker]["avg_price"] = new_avg_price
            self.portfolio[ticker]["last_buy"] = {
                "price": price,
                "quantity": quantity,
                "date": datetime.now().isoformat()
            }
        else:
            # Criar nova posição
            self.portfolio[ticker] = {
                "quantity": quantity,
                "avg_price": price,
                "last_buy": {
                    "price": price,
                    "quantity": quantity,
                    "date": datetime.now().isoformat()
                }
            }
        
        # Registrar transação
        transaction = {
            "ticker": ticker,
            "action": "BUY",
            "quantity": quantity,
            "price": price,
            "gross_value": gross_value,
            "commission": commission,
            "total_cost": total_cost,
            "timestamp": datetime.now().isoformat()
        }
        
        self.transaction_log.append(transaction)
        
        # Atualizar saldo
        self.account_balance -= total_cost
        
        return {
            "ticker": ticker,
            "action": "BUY",
            "success": True,
            "quantity": quantity,
            "price": price,
            "gross_value": gross_value,
            "commission": commission,
            "total_cost": total_cost,
            "new_quantity": self.portfolio[ticker]["quantity"],
            "new_avg_price": self.portfolio[ticker]["avg_price"],
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_max_affordable_quantity(self, price: float, available_balance: float) -> int:
        """Calcula a quantidade máxima que pode ser comprada com o saldo disponível."""
        if price <= 0:
            return 0
        
        # Começar com uma estimativa inicial
        initial_guess = int(available_balance / price)
        
        # Reduzir até encontrar a quantidade máxima que podemos pagar
        qty = initial_guess
        while qty > 0:
            cost = qty * price
            commission = calculate_xp_commission(cost)
            total = cost + commission
            
            if total <= available_balance:
                return qty
            
            qty -= 1
        
        return 0
    
    def _calculate_portfolio_summary(self) -> dict:
        """Calcula um resumo do portfólio atual."""
        total_invested = 0
        total_current_value = 0
        positions = []
        
        # Obter preços atuais
        tickers = list(self.portfolio.keys())
        current_prices = DataLoader.get_realtime_prices_bulk(tickers)
        
        for ticker, position in self.portfolio.items():
            quantity = position.get("quantity", 0)
            avg_price = position.get("avg_price", 0)
            
            invested_value = quantity * avg_price
            
            current_price = current_prices.get(ticker, avg_price)
            current_value = quantity * current_price
            
            profit_loss = current_value - invested_value
            profit_loss_pct = (profit_loss / invested_value * 100) if invested_value > 0 else 0
            
            positions.append({
                "ticker": ticker,
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "invested_value": invested_value,
                "current_value": current_value,
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct
            })
            
            total_invested += invested_value
            total_current_value += current_value
        
        return {
            "positions": positions,
            "total_invested": total_invested,
            "total_current_value": total_current_value,
            "total_profit_loss": total_current_value - total_invested,
            "total_profit_loss_pct": ((total_current_value / total_invested * 100) - 100) if total_invested > 0 else 0,
            "cash_balance": self.account_balance,
            "total_portfolio_value": total_current_value + self.account_balance,
            "position_count": len(positions),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_portfolio(self, file_path: str) -> bool:
        """Salva o portfólio atual em um arquivo."""
        data = {
            "portfolio": self.portfolio,
            "account_balance": self.account_balance,
            "timestamp": datetime.now().isoformat(),
            "transaction_log": self.transaction_log
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar portfólio: {e}")
            return False


class PortfolioOptimizer:
    """Otimiza uma carteira para maximizar potencial de retorno dentro dos limites de risco."""
    
    def __init__(self, portfolio: dict, watchlist: dict, account_balance: float, 
                 risk_profile: str = "medium", market_sentiment: dict = None):
        self.portfolio = portfolio
        self.watchlist = watchlist
        self.account_balance = account_balance
        self.risk_profile = risk_profile
        self.market_sentiment = market_sentiment or MarketAnalysis.get_market_sentiment()
        self.optimization_results = None
    
    def optimize(self, max_positions: int = 15, target_cash_reserve_pct: float = 0.1,
                max_position_size_pct: float = 0.2, min_score_threshold: float = 30) -> dict:
        """Executa otimização da carteira."""
        # 1. Analisar todas as posições atuais e potenciais
        portfolio_analysis = {}
        watchlist_analysis = {}
        
        # Obter preços atuais
        all_tickers = list(self.portfolio.keys()) + [t for t in self.watchlist.keys() if self.watchlist[t].get("monitor", False)]
        current_prices = DataLoader.get_realtime_prices_bulk(all_tickers)
        
        # Analisar posições atuais
        for ticker, position in self.portfolio.items():
            try:
                df = DataLoader.get_asset_data(ticker, days=60)
                if df.empty:
                    continue
                
                # Atualizar com preço atual
                current_price = current_prices.get(ticker)
                if current_price and 'Close' in df.columns:
                    df.iloc[-1, df.columns.get_loc("Close")] = current_price
                
                # Calcular score
                tech_result = Strategy.technical_score(
                    TechnicalIndicators.add_all_indicators(df, DYNAMIC_PARAMS),
                    self.risk_profile,
                    DYNAMIC_PARAMS
                )
                
                qual_result = QualitativeAnalysis.qualitative_score(ticker)
                fund_result = FundamentalAnalysis.fundamental_score(ticker)
                market_score = Strategy.market_adaptive_score(ticker, self.market_sentiment)
                
                total_score = (
                    DYNAMIC_PARAMS["weight_tech"] * tech_result["score"] +
                    DYNAMIC_PARAMS["weight_qual"] * qual_result["qualitative_score"] +
                    DYNAMIC_PARAMS["weight_fund"] * fund_result["fundamental_score"] +
                    DYNAMIC_PARAMS["weight_market"] * market_score
                )
                
                # Calcular performance atual
                qty = position.get("quantity", 0)
                avg_price = position.get("avg_price", 0)
                
                current_price = current_prices.get(ticker, avg_price)
                position_value = qty * current_price
                position_cost = qty * avg_price
                
                # Adicionar à análise
                portfolio_analysis[ticker] = {
                    "ticker": ticker,
                    "current_price": current_price,
                    "quantity": qty,
                    "avg_price": avg_price,
                    "position_value": position_value,
                    "position_cost": position_cost,
                    "pnl_pct": ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0,
                    "pnl_value": position_value - position_cost,
                    "score": total_score,
                    "tech_score": tech_result["score"],
                    "qual_score": qual_result["qualitative_score"],
                    "fund_score": fund_result["fundamental_score"],
                    "market_score": market_score
                }
                
            except Exception as e:
                logger.error(f"Erro ao analisar {ticker} para otimização: {e}")
        
        # Analisar watchlist
        for ticker, data in self.watchlist.items():
            if not data.get("monitor", False):
                continue
            
            try:
                df = DataLoader.get_asset_data(ticker, days=60)
                if df.empty:
                    continue
                
                # Atualizar com preço atual
                current_price = current_prices.get(ticker)
                if current_price and 'Close' in df.columns:
                    df.iloc[-1, df.columns.get_loc("Close")] = current_price
                
                # Calcular score
                tech_result = Strategy.technical_score(
                    TechnicalIndicators.add_all_indicators(df, DYNAMIC_PARAMS),
                    self.risk_profile,
                    DYNAMIC_PARAMS
                )
                
                qual_result = QualitativeAnalysis.qualitative_score(ticker)
                fund_result = FundamentalAnalysis.fundamental_score(ticker)
                market_score = Strategy.market_adaptive_score(ticker, self.market_sentiment)
                
                total_score = (
                    DYNAMIC_PARAMS["weight_tech"] * tech_result["score"] +
                    DYNAMIC_PARAMS["weight_qual"] * qual_result["qualitative_score"] +
                    DYNAMIC_PARAMS["weight_fund"] * fund_result["fundamental_score"] +
                    DYNAMIC_PARAMS["weight_market"] * market_score
                )
                
                # Adicionar à análise se score for adequado
                if total_score >= min_score_threshold:
                    watchlist_analysis[ticker] = {
                        "ticker": ticker,
                        "current_price": current_prices.get(ticker, 0),
                        "score": total_score,
                        "tech_score": tech_result["score"],
                        "qual_score": qual_result["qualitative_score"],
                        "fund_score": fund_result["fundamental_score"],
                        "market_score": market_score
                    }
                
            except Exception as e:
                logger.error(f"Erro ao analisar watchlist {ticker} para otimização: {e}")
        
        # 2. Construir carteira ideal
        # Calcular valor total do portfólio
        total_portfolio_value = sum(pos["position_value"] for pos in portfolio_analysis.values()) + self.account_balance
        
        # Determinar alocação de capital ideal
        target_cash_reserve = total_portfolio_value * target_cash_reserve_pct
        investable_capital = total_portfolio_value - target_cash_reserve
        target_position_value = investable_capital / max_positions
        max_position_value = total_portfolio_value * max_position_size_pct
        
        # Classificar ativos por score
        all_assets = []
        
        # Adicionar posições atuais
        for ticker, analysis in portfolio_analysis.items():
            all_assets.append({
                "ticker": ticker,
                "score": analysis["score"],
                "current_position_value": analysis["position_value"],
                "current_position_qty": analysis["quantity"],
                "price": analysis["current_price"],
                "avg_price": analysis["avg_price"],
                "current_allocation_pct": analysis["position_value"] / total_portfolio_value * 100 if total_portfolio_value > 0 else 0,
                "in_portfolio": True
            })
        
        # Adicionar posições potenciais da watchlist
        for ticker, analysis in watchlist_analysis.items():
            if ticker not in portfolio_analysis:
                all_assets.append({
                    "ticker": ticker,
                    "score": analysis["score"],
                    "current_position_value": 0,
                    "current_position_qty": 0,
                    "price": analysis["current_price"],
                    "avg_price": 0,
                    "current_allocation_pct": 0,
                    "in_portfolio": False
                })
        
        # Ordenar por score (maior para menor)
        all_assets.sort(key=lambda x: x["score"], reverse=True)
        
        # 3. Determinar mudanças necessárias
        target_portfolio = []
        assets_to_buy = []
        assets_to_sell = []
        assets_to_hold = []
        
        # Selecionar os top assets com melhores scores até max_positions
        top_assets = all_assets[:max_positions]
        
        # Calcular alocação ideal
        for asset in top_assets:
            target_value = min(target_position_value, max_position_value)
            
            # Calcular ações necessárias (compra, venda ou manter)
            current_value = asset["current_position_value"]
            difference = target_value - current_value
            
            # Se já temos uma posição
            if asset["in_portfolio"]:
                if difference > 0:
                    # Comprar mais
                    qty_to_buy = int(difference / asset["price"]) if asset["price"] > 0 else 0
                    
                    if qty_to_buy > 0:
                        assets_to_buy.append({
                            "ticker": asset["ticker"],
                            "buy_qty": qty_to_buy,
                            "target_value": target_value,
                            "current_value": current_value,
                            "price": asset["price"],
                            "score": asset["score"]
                        })
                elif difference < -50:  # Só se preocupar com ajustes significativos
                    # Vender parcialmente
                    qty_to_sell = int(abs(difference) / asset["price"]) if asset["price"] > 0 else 0
                    
                    if qty_to_sell > 0 and qty_to_sell < asset["current_position_qty"]:
                        assets_to_sell.append({
                            "ticker": asset["ticker"],
                            "sell_qty": qty_to_sell,
                            "target_value": target_value,
                            "current_value": current_value,
                            "price": asset["price"],
                            "score": asset["score"]
                        })
                    else:
                        assets_to_hold.append({
                            "ticker": asset["ticker"],
                            "qty": asset["current_position_qty"],
                            "target_value": target_value,
                            "current_value": current_value,
                            "price": asset["price"],
                            "score": asset["score"]
                        })
                else:
                    # Manter
                    assets_to_hold.append({
                        "ticker": asset["ticker"],
                        "qty": asset["current_position_qty"],
                        "target_value": target_value,
                        "current_value": current_value,
                        "price": asset["price"],
                        "score": asset["score"]
                    })
            else:
                # Novo ativo para comprar
                qty_to_buy = int(target_value / asset["price"]) if asset["price"] > 0 else 0
                
                if qty_to_buy > 0:
                    assets_to_buy.append({
                        "ticker": asset["ticker"],
                        "buy_qty": qty_to_buy,
                        "target_value": target_value,
                        "current_value": 0,
                        "price": asset["price"],
                        "score": asset["score"]
                    })
            
            # Adicionar à carteira alvo
            target_portfolio.append({
                "ticker": asset["ticker"],
                "target_value": target_value,
                "target_allocation_pct": target_value / total_portfolio_value * 100 if total_portfolio_value > 0 else 0,
                "current_value": asset["current_position_value"],
                "current_allocation_pct": asset["current_allocation_pct"],
                "difference": difference,
                "score": asset["score"]
            })
        
        # 4. Identificar ativos que devem ser totalmente vendidos (não estão na carteira ideal)
        for asset in all_assets:
            if asset["in_portfolio"] and not any(t["ticker"] == asset["ticker"] for t in top_assets):
                assets_to_sell.append({
                    "ticker": asset["ticker"],
                    "sell_qty": asset["current_position_qty"],
                    "target_value": 0,
                    "current_value": asset["current_position_value"],
                    "price": asset["price"],
                    "score": asset["score"],
                    "reason": "Não está entre os melhores ativos"
                })
        
        # 5. Gerar resultado da otimização
        self.optimization_results = {
            "target_portfolio": target_portfolio,
            "assets_to_buy": assets_to_buy,
            "assets_to_sell": assets_to_sell,
            "assets_to_hold": assets_to_hold,
            "total_portfolio_value": total_portfolio_value,
            "total_current_invested": sum(pos["position_value"] for pos in portfolio_analysis.values()),
            "total_target_invested": sum(asset["target_value"] for asset in target_portfolio),
            "current_cash": self.account_balance,
            "target_cash_reserve": target_cash_reserve,
            "buy_value": sum(buy["buy_qty"] * buy["price"] for buy in assets_to_buy),
            "sell_value": sum(sell["sell_qty"] * sell["price"] for sell in assets_to_sell),
            "max_positions": max_positions,
            "target_cash_reserve_pct": target_cash_reserve_pct,
            "max_position_size_pct": max_position_size_pct,
            "positions_count": len(target_portfolio),
            "timestamp": datetime.now().isoformat()
        }
        
        return self.optimization_results
    
    def generate_rebalance_plan(self) -> dict:
        """Gera um plano de rebalanceamento com base na otimização."""
        if not self.optimization_results:
            raise Exception("Execute o método optimize() primeiro")
        
        rebalance_plan = {"sell": [], "buy": [], "rebalance": [], "stats": {}}
        
        # Converter vendas da otimização para plano de rebalanceamento
        for sell in self.optimization_results["assets_to_sell"]:
            operation_value = sell["sell_qty"] * sell["price"]
            commission = calculate_xp_commission(operation_value)
            
            rebalance_plan["sell"].append({
                "ticker": sell["ticker"],
                "sell_quantity": sell["sell_qty"],
                "current_price": sell["price"],
                "operation_value": operation_value,
                "commission": commission,
                "capital_freed": operation_value - commission,
                "reason": sell.get("reason", "Otimização: Ajuste de alocação"),
                "score": sell["score"]
            })
        
        # Converter compras da otimização para plano de rebalanceamento
        for buy in self.optimization_results["assets_to_buy"]:
            operation_value = buy["buy_qty"] * buy["price"]
            commission = calculate_xp_commission(operation_value)
            
            rebalance_plan["buy"].append({
                "ticker": buy["ticker"],
                "buy_quantity": buy["buy_qty"],
                "current_price": buy["price"],
                "operation_value": operation_value,
                "commission": commission,
                "total_cost": operation_value + commission,
                "reason": f"Otimização: Score {buy['score']:.2f}",
                "score": buy["score"]
            })
        
        # Estatísticas do plano
        capital_from_sales = sum(item["capital_freed"] for item in rebalance_plan["sell"])
        capital_for_buys = sum(item["total_cost"] for item in rebalance_plan["buy"])
        
        rebalance_plan["stats"] = {
            "saldo_inicial": self.account_balance,
            "capital_liberado_vendas": capital_from_sales,
            "capital_total_disponivel": self.account_balance + capital_from_sales,
            "capital_usado_compras": capital_for_buys,
            "saldo_remanescente": self.account_balance + capital_from_sales - capital_for_buys,
            "total_comissoes": sum(s["commission"] for s in rebalance_plan["sell"]) + sum(b["commission"] for b in rebalance_plan["buy"]),
            "numero_vendas": len(rebalance_plan["sell"]),
            "numero_rebalanceamentos": len(rebalance_plan["rebalance"]),
            "numero_compras": len(rebalance_plan["buy"])
        }
        
        return rebalance_plan

# =============================================================================
# Constantes de versão e configuração
# =============================================================================
VERSION = "2.0.0"
SYSTEM_NAME = "Sistema de Trading Swing Trade Agressivo"
SYSTEM_AUTHOR = "Equipe de Trading"
BUILD_DATE = "2025-03-09"

# =============================================================================
# Execução principal (main)
# =============================================================================
def main():
    """Função principal para execução do sistema de trading."""
    # Parsear argumentos de linha de comando
    args = parse_command_line_args()
    
    # Configuração de logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Modo de debug ativado")
    
    # Verificar versão se solicitado
    if args.version:
        print(f"{SYSTEM_NAME} - Versão {VERSION}")
        print(f"Desenvolvido por: {SYSTEM_AUTHOR}")
        print(f"Build: {BUILD_DATE}")
        return
    
    # Executar diagnósticos se solicitado
    if args.diagnostics:
        print("Executando diagnósticos do sistema...")
        diagnostics = run_diagnostics()
        print(json.dumps(diagnostics, indent=2, default=str))
        return
    
    # Carregar arquivo de configuração adicional se fornecido
    config = {}
    if args.config:
        config = load_configuration(args.config)
        setup_logging(config.get("logging", {}))
    
    # Verificar se arquivo JSON foi fornecido
    if not args.json_file:
        print("Por favor, especifique o arquivo JSON de configuração usando --json-file")
        if getattr(args, "interactive", False):
            interactive_mode()
            return
        else:
            sys.exit(1)
    
    # Carregar dados do arquivo JSON
    data = load_portfolio(args.json_file)
    if not data:
        print(f"Erro ao carregar dados do arquivo: {args.json_file}")
        sys.exit(1)
    
    # Extrair componentes do arquivo
    portfolio = data.get("portfolio", {})
    watchlist = data.get("watchlist", {})
    account_balance = data.get("account_balance", 0.0)
    goals = data.get("goals", {})
    
    if args.debug:
        print(f"Dados carregados: {len(portfolio)} ativos em carteira, {len(watchlist)} ativos em watchlist")
        print(f"Saldo: ${account_balance:,.2f}")
        if goals:
            print(f"Meta de recuperação: ${goals.get('target_recovery', 0):,.2f} em {goals.get('days', 0)} dias")
    
    # Validar tickers se solicitado
    if args.validate_tickers:
        print("Validando tickers...")
        invalid_portfolio = []
        invalid_watchlist = []
        
        for ticker in portfolio.keys():
            if not DataLoader.check_ticker_valid(ticker):
                invalid_portfolio.append(ticker)
                logger.warning(f"Ticker inválido em carteira: {ticker}")
        
        for ticker in watchlist.keys():
            if watchlist[ticker].get("monitor", False) and not DataLoader.check_ticker_valid(ticker):
                invalid_watchlist.append(ticker)
                logger.warning(f"Ticker inválido em watchlist: {ticker}")
        
        if invalid_portfolio:
            print(f"Tickers inválidos em carteira: {', '.join(invalid_portfolio)}")
        
        if invalid_watchlist:
            print(f"Tickers inválidos em watchlist: {', '.join(invalid_watchlist)}")
    
    # Criar sistema de notificações
    notification = NotificationSystem(config.get("notifications", {}))
    
    # Inicializar dados de trailing stop
    trailing_data = {}
    
    # Executar otimização da carteira se solicitado
    if args.optimize:
        print("Otimizando carteira...")
        start_time = datetime.now()
        
        market_sentiment = MarketAnalysis.get_market_sentiment()
        optimizer = PortfolioOptimizer(
            portfolio, watchlist, account_balance, args.risk, market_sentiment
        )
        
        optimization_results = optimizer.optimize()
        rebalance_plan = optimizer.generate_rebalance_plan()
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        if args.debug:
            print(f"Otimização concluída em {elapsed:.2f} segundos")
        
        # Exibir resultados da otimização
        if args.output_format == "json":
            print(json.dumps(optimization_results, indent=2, default=str))
        else:
            print("\n===== RESULTADO DA OTIMIZAÇÃO =====")
            print(f"Valor total da carteira: ${optimization_results['total_portfolio_value']:,.2f}")
            print(f"Saldo atual: ${optimization_results['current_cash']:,.2f}")
            print(f"Reserva de caixa alvo: ${optimization_results['target_cash_reserve']:,.2f}")
            
            print("\nAtivos a manter:")
            for asset in optimization_results['assets_to_hold']:
                print(f"- {asset['ticker']}: {asset['qty']} ações, ${asset['current_value']:,.2f}")
            
            print("\nAtivos a vender:")
            for asset in optimization_results['assets_to_sell']:
                print(f"- {asset['ticker']}: {asset['sell_qty']} ações, ${asset['price']:,.2f}/ação")
            
            print("\nAtivos a comprar:")
            for asset in optimization_results['assets_to_buy']:
                print(f"- {asset['ticker']}: {asset['buy_qty']} ações, ${asset['price']:,.2f}/ação")
        
        # Exportar resultados se solicitado
        if args.export_file:
            if export_results(optimization_results, args.export_file, args.export_format):
                print(f"Resultados exportados para: {args.export_file}")
        
        return
    
    # Executar backtest se solicitado
    if args.backtest:
        print("Executando backtest...")
        
        # Criar histórico simulado (na prática seria carregado de arquivo)
        # Esta é apenas uma simulação usando preços atuais
        portfolio_history = []
        
        # Obter valores atuais para simular histórico
        current_values = {}
        for ticker, position in portfolio.items():
            df = DataLoader.get_asset_data(ticker, days=args.backtest_days)
            if not df.empty and 'Close' in df.columns:
                current_values[ticker] = position.get("quantity", 0) * df['Close'].iloc[-1]
        
        total_value = sum(current_values.values()) + account_balance
        
        # Adicionar snapshot inicial
        portfolio_history.append({
            "date": (datetime.now() - timedelta(days=args.backtest_days)).strftime("%Y-%m-%d"),
            "total_value": total_value,
            "cash": account_balance,
            "positions": current_values
        })
        
        # Adicionar snapshot final (atual)
        portfolio_history.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_value": total_value,
            "cash": account_balance,
            "positions": current_values
        })
        
        # Executar backtest
        benchmark = args.backtest_benchmark if hasattr(args, "backtest_benchmark") else "SPY"
        backtest_results = create_backtest(portfolio_history, benchmark, args.backtest_days)
        
        # Exibir resultados
        if args.output_format == "json":
            print(json.dumps(backtest_results, indent=2, default=str))
        else:
            if "error" in backtest_results:
                print(f"Erro no backtest: {backtest_results['error']}")
            else:
                print("\n===== RESULTADOS DO BACKTEST =====")
                
                portfolio_metrics = backtest_results.get("portfolio_metrics", {})
                benchmark_metrics = backtest_results.get("benchmark_metrics", {})
                
                print(f"Período: {backtest_results.get('period_days', 0)} dias")
                print(f"Benchmark: {backtest_results.get('benchmark_ticker', '')}")
                
                print(f"\nRetorno total da carteira: {portfolio_metrics.get('total_return_pct', 0):.2f}%")
                print(f"Retorno anualizado: {portfolio_metrics.get('annual_return_pct', 0):.2f}%")
                print(f"Volatilidade: {portfolio_metrics.get('volatility_pct', 0):.2f}%")
                print(f"Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")
                print(f"Máximo Drawdown: {portfolio_metrics.get('max_drawdown_pct', 0):.2f}%")
                
                print(f"\nRetorno total do benchmark: {benchmark_metrics.get('total_return_pct', 0):.2f}%")
        
        # Exportar resultados se solicitado
        if args.export_file:
            if export_results(backtest_results, args.export_file, args.export_format):
                print(f"Resultados exportados para: {args.export_file}")
        
        return
    
   # Analisar portfólio
    print("Analisando carteira...")
    start_time = datetime.now()
    
    portfolio_analysis = analyze_portfolio(
        portfolio, account_balance, args.risk,
        trailing_data, extended_hours=args.extended, goals=goals
    )
    
    # Analisar watchlist
    print("Analisando watchlist...")
    watchlist_analysis = analyze_watchlist(
        watchlist, account_balance, args.risk,
        extended_hours=args.extended, goals=goals
    )
    
    # Gerar plano de rebalanceamento
    print("Gerando plano de rebalanceamento...")
    rebalance_plan = generate_rebalance_plan(
        portfolio_analysis, watchlist_analysis,
        account_balance, DYNAMIC_PARAMS
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    if args.debug:
        print(f"Análise concluída em {elapsed:.2f} segundos")
    
    # Simular execução se solicitado
    if args.simulate:
        print("\nSimulando execução do plano de rebalanceamento...")
        executor = PortfolioExecutor(portfolio, account_balance)
        simulation_results = executor.execute_rebalance_plan(rebalance_plan)
        
        if args.output_format == "json":
            print(json.dumps(simulation_results, indent=2, default=str))
        else:
            print("\n===== RESULTADO DA SIMULAÇÃO =====")
            print(f"Saldo inicial: ${simulation_results['starting_balance']:,.2f}")
            print(f"Capital liberado: ${simulation_results['capital_freed']:,.2f}")
            print(f"Capital investido: ${simulation_results['capital_invested']:,.2f}")
            print(f"Saldo final: ${simulation_results['ending_balance']:,.2f}")
            print(f"Comissões totais: ${simulation_results['total_commission']:,.2f}")
            
            print(f"\nOperações executadas: {simulation_results['executed_transactions']}")
            print(f"- Vendas: {len(simulation_results['sells_executed'])}")
            print(f"- Compras: {len(simulation_results['buys_executed'])}")
            
            if simulation_results['errors']:
                print(f"\nErros: {len(simulation_results['errors'])}")
                for error in simulation_results['errors']:
                    print(f"- {error.get('ticker', 'Unknown')}: {error.get('error', 'Unknown error')}")
        
        # Exportar resultados da simulação se solicitado
        if args.export_file:
            if export_results(simulation_results, args.export_file, args.export_format):
                print(f"Resultados da simulação exportados para: {args.export_file}")
        
        return
    
    # Exibir resultados
    if args.output_format == "json":
        combined_results = {
            "portfolio_analysis": portfolio_analysis,
            "watchlist_analysis": watchlist_analysis,
            "rebalance_plan": rebalance_plan,
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": elapsed
        }
        print(json.dumps(combined_results, indent=2, default=str))
    elif args.output_format == "table" and HAS_TABULATE:
        output_table_results(portfolio_analysis, watchlist_analysis, rebalance_plan)
    else:
        output_text_results(portfolio_analysis, watchlist_analysis, rebalance_plan)
    
    # Exportar resultados se solicitado
    if args.export_file:
        export_data = {
            "portfolio_analysis": portfolio_analysis,
            "watchlist_analysis": watchlist_analysis,
            "rebalance_plan": rebalance_plan,
            "timestamp": datetime.now().isoformat()
        }
        
        if export_results(export_data, args.export_file, args.export_format):
            print(f"Resultados exportados para: {args.export_file}")

# Função para saída em formato de texto
def output_text_results(portfolio_analysis, watchlist_analysis, rebalance_plan):
    """Exibe os resultados da análise em formato de texto."""
    # Resumo da carteira
    print("\n" + "="*50)
    print("RESUMO DA ANÁLISE DE CARTEIRA")
    print("="*50)
    
    resumo = portfolio_analysis.get("resumo", {})
    
    print(f"Valor total da carteira: ${resumo.get('valor_total', 0):,.2f}")
    print(f"Saldo em conta: ${resumo.get('saldo_conta', 0):,.2f}")
    print(f"Total investido: ${resumo.get('valor_investido', 0):,.2f}")
    print(f"Lucro/Prejuízo: ${resumo.get('lucro_prejuizo_valor', 0):,.2f} ({resumo.get('lucro_prejuizo_pct', 0):.2f}%)")
    print(f"Número de posições: {resumo.get('numero_posicoes', 0)}")
    print(f"Score médio da carteira: {resumo.get('score_medio', 0):.2f}")
    
    # Posições analisadas
    print("\n" + "-"*50)
    print("POSIÇÕES ANALISADAS")
    print("-"*50)
    
    for ticker, details in portfolio_analysis.get("ativos", {}).items():
        if "error" in details:
            print(f"\n* {ticker}: ERRO - {details['error']}")
            continue
        
        print(f"\n* {ticker}:")
        print(f"  Quantidade: {details.get('quantity', 0)}")
        print(f"  Preço atual: ${details.get('current_price', 0):,.2f}")
        print(f"  Valor da posição: ${details.get('position_value', 0):,.2f}")
        print(f"  Lucro/Prejuízo: ${details.get('position_profit_value', 0):,.2f} ({details.get('position_profit_pct', 0):.2f}%)")
        print(f"  Score: {details.get('total_score', 0):.2f}")
        print(f"  Decisão: {details.get('decision', 'AGUARDAR')}")
        print(f"  Justificativa: {details.get('justificativa', '')}")
    
    # Plano de rebalanceamento
    print("\n" + "="*50)
    print("PLANO DE REBALANCEAMENTO")
    print("="*50)
    
    stats = rebalance_plan.get("stats", {})
    
    print(f"Saldo inicial: ${stats.get('saldo_inicial', 0):,.2f}")
    print(f"Capital liberado com vendas: ${stats.get('capital_liberado_vendas', 0):,.2f}")
    print(f"Capital total disponível: ${stats.get('capital_total_disponivel', 0):,.2f}")
    print(f"Capital usado em compras: ${stats.get('capital_usado_compras', 0):,.2f}")
    print(f"Saldo remanescente: ${stats.get('saldo_remanescente', 0):,.2f}")
    print(f"Total de comissões: ${stats.get('total_comissoes', 0):,.2f}")
    
    # Vendas recomendadas
    if rebalance_plan.get("sell"):
        print("\n" + "-"*50)
        print("VENDAS RECOMENDADAS")
        print("-"*50)
        
        for sell in rebalance_plan["sell"]:
            print(f"\n* Vender {sell.get('sell_quantity', 0)} ação(ões) de {sell.get('ticker', '')}")
            print(f"  Preço atual: ${sell.get('current_price', 0):,.2f}")
            print(f"  Valor da operação: ${sell.get('operation_value', 0):,.2f}")
            print(f"  Comissão: ${sell.get('commission', 0):,.2f}")
            print(f"  Capital liberado: ${sell.get('capital_freed', 0):,.2f}")
            print(f"  Razão: {sell.get('reason', '')}")
    
    # Rebalanceamentos recomendados
    if rebalance_plan.get("rebalance"):
        print("\n" + "-"*50)
        print("REBALANCEAMENTOS RECOMENDADOS")
        print("-"*50)
        
        for rebalance in rebalance_plan["rebalance"]:
            print(f"\n* Reduzir {rebalance.get('sell_quantity', 0)} ação(ões) de {rebalance.get('ticker', '')}")
            print(f"  Preço atual: ${rebalance.get('current_price', 0):,.2f}")
            print(f"  Valor da operação: ${rebalance.get('operation_value', 0):,.2f}")
            print(f"  Comissão: ${rebalance.get('commission', 0):,.2f}")
            print(f"  Capital liberado: ${rebalance.get('capital_freed', 0):,.2f}")
            print(f"  Razão: {rebalance.get('reason', '')}")
    
    # Compras recomendadas
    if rebalance_plan.get("buy"):
        print("\n" + "-"*50)
        print("COMPRAS RECOMENDADAS")
        print("-"*50)
        
        for buy in rebalance_plan["buy"]:
            print(f"\n* Comprar {buy.get('buy_quantity', 0)} ação(ões) de {buy.get('ticker', '')}")
            print(f"  Preço atual: ${buy.get('current_price', 0):,.2f}")
            print(f"  Valor da operação: ${buy.get('operation_value', 0):,.2f}")
            print(f"  Comissão: ${buy.get('commission', 0):,.2f}")
            print(f"  Custo total: ${buy.get('total_cost', 0):,.2f}")
            print(f"  Razão: {buy.get('reason', '')}")

# Função para saída em formato de tabela
def output_table_results(portfolio_analysis, watchlist_analysis, rebalance_plan):
    """Exibe os resultados da análise em formato de tabela."""
    from tabulate import tabulate
    
    # Resumo da carteira
    print("\n" + "="*50)
    print("RESUMO DA ANÁLISE DE CARTEIRA")
    print("="*50)
    
    resumo = portfolio_analysis.get("resumo", {})
    resumo_table = [
        ["Valor total da carteira", f"${resumo.get('valor_total', 0):,.2f}"],
        ["Saldo em conta", f"${resumo.get('saldo_conta', 0):,.2f}"],
        ["Total investido", f"${resumo.get('valor_investido', 0):,.2f}"],
        ["Lucro/Prejuízo", f"${resumo.get('lucro_prejuizo_valor', 0):,.2f} ({resumo.get('lucro_prejuizo_pct', 0):.2f}%)"],
        ["Número de posições", resumo.get('numero_posicoes', 0)],
        ["Score médio da carteira", f"{resumo.get('score_medio', 0):.2f}"]
    ]
    
    print(tabulate(resumo_table, tablefmt="grid"))
    
    # Posições analisadas
    print("\n" + "="*50)
    print("POSIÇÕES ANALISADAS")
    print("="*50)
    
    positions_table = []
    
    for ticker, details in portfolio_analysis.get("ativos", {}).items():
        if "error" in details:
            positions_table.append([
                ticker,
                "ERRO",
                0,
                "$0.00",
                "$0.00",
                "$0.00 (0.00%)",
                "0.00",
                "ERRO",
                details['error']
            ])
            continue
        
        positions_table.append([
            ticker,
            details.get("quantity", 0),
            f"${details.get('current_price', 0):,.2f}",
            f"${details.get('position_value', 0):,.2f}",
            f"${details.get('position_profit_value', 0):,.2f} ({details.get('position_profit_pct', 0):.2f}%)",
            f"{details.get('total_score', 0):.2f}",
            details.get('decision', 'AGUARDAR'),
            details.get('justificativa', '')
        ])
    
    headers = ["Ticker", "Qtde", "Preço", "Valor", "Lucro/Prejuízo", "Score", "Decisão", "Justificativa"]
    print(tabulate(positions_table, headers=headers, tablefmt="grid"))
    
    # Plano de rebalanceamento
    print("\n" + "="*50)
    print("PLANO DE REBALANCEAMENTO")
    print("="*50)
    
    stats = rebalance_plan.get("stats", {})
    stats_table = [
        ["Saldo inicial", f"${stats.get('saldo_inicial', 0):,.2f}"],
        ["Capital liberado com vendas", f"${stats.get('capital_liberado_vendas', 0):,.2f}"],
        ["Capital total disponível", f"${stats.get('capital_total_disponivel', 0):,.2f}"],
        ["Capital usado em compras", f"${stats.get('capital_usado_compras', 0):,.2f}"],
        ["Saldo remanescente", f"${stats.get('saldo_remanescente', 0):,.2f}"],
        ["Total de comissões", f"${stats.get('total_comissoes', 0):,.2f}"]
    ]
    
    print(tabulate(stats_table, tablefmt="grid"))
    
    # Vendas recomendadas
    if rebalance_plan.get("sell"):
        print("\n" + "="*50)
        print("VENDAS RECOMENDADAS")
        print("="*50)
        
        sell_table = []
        
        for sell in rebalance_plan["sell"]:
            sell_table.append([
                sell.get('ticker', ''),
                sell.get('sell_quantity', 0),
                f"${sell.get('current_price', 0):,.2f}",
                f"${sell.get('operation_value', 0):,.2f}",
                f"${sell.get('commission', 0):,.2f}",
                f"${sell.get('capital_freed', 0):,.2f}",
                sell.get('reason', '')
            ])
        
        headers = ["Ticker", "Qtde", "Preço", "Valor", "Comissão", "Capital Liberado", "Razão"]
        print(tabulate(sell_table, headers=headers, tablefmt="grid"))
    
    # Rebalanceamentos recomendados
    if rebalance_plan.get("rebalance"):
        print("\n" + "="*50)
        print("REBALANCEAMENTOS RECOMENDADOS")
        print("="*50)
        
        rebalance_table = []
        
        for rebalance in rebalance_plan["rebalance"]:
            rebalance_table.append([
                rebalance.get('ticker', ''),
                rebalance.get('sell_quantity', 0),
                f"${rebalance.get('current_price', 0):,.2f}",
                f"${rebalance.get('operation_value', 0):,.2f}",
                f"${rebalance.get('commission', 0):,.2f}",
                f"${rebalance.get('capital_freed', 0):,.2f}",
                rebalance.get('reason', '')
            ])
        
        headers = ["Ticker", "Qtde", "Preço", "Valor", "Comissão", "Capital Liberado", "Razão"]
        print(tabulate(rebalance_table, headers=headers, tablefmt="grid"))
    
    # Compras recomendadas
    if rebalance_plan.get("buy"):
        print("\n" + "="*50)
        print("COMPRAS RECOMENDADAS")
        print("="*50)
        
        buy_table = []
        
        for buy in rebalance_plan["buy"]:
            buy_table.append([
                buy.get('ticker', ''),
                buy.get('buy_quantity', 0),
                f"${buy.get('current_price', 0):,.2f}",
                f"${buy.get('operation_value', 0):,.2f}",
                f"${buy.get('commission', 0):,.2f}",
                f"${buy.get('total_cost', 0):,.2f}",
                buy.get('reason', '')
            ])
        
        headers = ["Ticker", "Qtde", "Preço", "Valor", "Comissão", "Custo Total", "Razão"]
        print(tabulate(buy_table, headers=headers, tablefmt="grid"))

# Executar o programa se executado diretamente
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperação cancelada pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\nErro não tratado: {e}")
        logger.error(f"Erro não tratado na execução principal: {e}", exc_info=True)
        sys.exit(1)