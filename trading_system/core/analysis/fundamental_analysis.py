import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger("trading_system.fundamental_analysis")

class FundamentalAnalysis:
    """Análise fundamental com yfinance."""
    
    @staticmethod
    def fundamental_score(ticker: str) -> Dict[str, Any]:
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