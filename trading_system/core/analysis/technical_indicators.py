import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger("trading_system.technical_indicators")

class TechnicalIndicators:
    """Cálculo dos indicadores técnicos."""
    
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
            
            if 'Volume' in df_ind.columns:
                df_ind['OBV'] = 0  # Simplified placeholder for OBV
                df_ind['OBV_SMA'] = TechnicalIndicators.SMA(df_ind['OBV'], 10)
        
        except Exception as e:
            logger.error(f"Erro ao adicionar indicadores técnicos: {e}")
        
        return df_ind