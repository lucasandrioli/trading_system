import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
from typing import Dict, List, Optional

# Configure logging
logger = logging.getLogger("trading_system.data_loader")

class DataLoader:
    """Handles loading market data from various sources."""
    
    POLYGON_API_KEY = None
    
    @staticmethod
    def get_asset_data(ticker: str, days: int = 60, interval: str = "1d",
                       use_polygon: bool = False, extended_hours: bool = False) -> pd.DataFrame:
        """Obtém dados históricos para um ticker específico."""
        try:
            if not ticker or not isinstance(ticker, str):
                raise ValueError(f"Ticker inválido: {ticker}")
            
            # Tentar primeiro com yfinance (mais confiável)
            df = DataLoader._get_asset_data_yfinance(ticker, days, interval, include_prepost=extended_hours)
            
            # Se yfinance falhar e Polygon estiver configurado, tente Polygon como fallback
            if (df is None or df.empty) and use_polygon and DataLoader.POLYGON_API_KEY != "YOUR_API_KEY_HERE":
                logger.info(f"Dados não encontrados via yfinance para {ticker}, tentando Polygon...")
                df = DataLoader._get_asset_data_polygon(ticker, days, interval)
            
            if df is None or df.empty:
                logger.warning(f"Dados vazios retornados para {ticker}")
                return pd.DataFrame()
            
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar dados para {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _get_asset_data_yfinance(ticker: str, days: int = 60, interval: str = "1d",
                                 include_prepost: bool = False) -> pd.DataFrame:
        period = f"{days}d"
        ticker_obj = yf.Ticker(ticker)
        try:
            df = ticker_obj.history(period=period, interval=interval, auto_adjust=True, prepost=include_prepost)
            
            if df.empty:
                logger.warning(f"Nenhum dado retornado do yfinance para {ticker}.")
                return pd.DataFrame()
            
            # Padronização dos nomes das colunas
            df = df.rename(columns=str.title)
            
            # Garante que a coluna Close existe e adiciona retorno diário
            if 'Close' in df.columns:
                df['Daily_Return'] = df['Close'].pct_change()
            
            # Processamento do índice/data
            df = df.reset_index()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif 'Datetime' in df.columns:
                df['Date'] = pd.to_datetime(df['Datetime'])
                df.drop(columns=['Datetime'], inplace=True)
            
            df = df.sort_values('Date').set_index('Date')
            return df
        except Exception as e:
            logger.error(f"Erro ao obter dados via yfinance para {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _get_asset_data_polygon(ticker: str, days: int = 60, interval: str = "1d") -> pd.DataFrame:
        if not DataLoader.POLYGON_API_KEY:
            return pd.DataFrame()
            
        try:
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            timespan_map = {"1d": "day", "1h": "hour", "1m": "minute"}
            api_timespan = timespan_map.get(interval, "day")
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{api_timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apiKey": DataLoader.POLYGON_API_KEY, "limit": 5000, "adjusted": "true", "sort": "asc"}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or "results" not in data:
                logger.warning(f"Nenhum dado encontrado na Polygon para {ticker}.")
                return pd.DataFrame()
            
            df = pd.DataFrame(data["results"])
            if df.empty:
                return pd.DataFrame()
            
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                "o": "Open", "h": "High", "l": "Low",
                "c": "Close", "v": "Volume", "n": "Transactions", "vw": "VWAP"
            })
            
            df = df.sort_values('Date').set_index('Date')
            df.drop(columns=['t'], errors='ignore', inplace=True)
            
            if 'Close' in df.columns:
                df['Daily_Return'] = df['Close'].pct_change()
            
            return df
        except Exception as e:
            logger.error(f"Erro na chamada Polygon para {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_realtime_prices_bulk(tickers: list) -> dict:
        """Obtém preços em tempo real para múltiplos tickers."""
        prices = {}
        if not tickers:
            return prices
        
        # Dividir tickers em lotes menores para evitar falhas com muitos tickers
        max_batch_size = 50
        ticker_batches = [tickers[i:i + max_batch_size] for i in range(0, len(tickers), max_batch_size)]
        
        for batch in ticker_batches:
            try:
                batch_prices = DataLoader._get_batch_prices(batch)
                prices.update(batch_prices)
            except Exception as e:
                logger.error(f"Erro ao obter preços para lote de tickers: {e}")
        
        # Verificar tickers que não foram obtidos e tentar individualmente
        missing = [t for t in tickers if t not in prices]
        for t in missing:
            try:
                hist = yf.Ticker(t).history(period="1d", auto_adjust=True)
                if not hist.empty and 'Close' in hist.columns:
                    latest_close = hist['Close'].iloc[-1]
                    if not pd.isna(latest_close):
                        prices[t] = float(latest_close)
            except Exception as e:
                logger.error(f"Não foi possível obter preço para {t}: {e}")
        
        return prices

    @staticmethod
    def _get_batch_prices(tickers: list) -> dict:
        """Método auxiliar para obter preços em lote."""
        prices = {}
        if not tickers:
            return prices
        
        try:
            tickers_str = " ".join(tickers)
            data = yf.download(tickers_str, period="1d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
            
            if len(tickers) == 1 and not data.empty:
                t = tickers[0]
                if 'Close' in data.columns:
                    prices[t] = float(data['Close'].iloc[-1])
            elif len(tickers) > 1:
                for t in tickers:
                    try:
                        if t in data and not data[t].empty and 'Close' in data[t].columns:
                            latest_close = data[t]['Close'].iloc[-1]
                            if not pd.isna(latest_close):
                                prices[t] = float(latest_close)
                    except Exception as e:
                        logger.error(f"Erro ao processar preço para {t}: {e}")
        except Exception as e:
            logger.error(f"Erro geral ao obter preços em lote: {e}")
        
        return prices

    @staticmethod
    def check_ticker_valid(ticker: str) -> bool:
        """Verifica se um ticker é válido e tem dados disponíveis."""
        try:
            ticker_obj = yf.Ticker(ticker)