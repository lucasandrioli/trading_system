import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from trading_system.core.analysis.technical_indicators import TechnicalIndicators
from trading_system.core.analysis.fundamental_analysis import FundamentalAnalysis
from trading_system.core.analysis.market_analysis import MarketAnalysis
from trading_system.core.strategy.strategy import Strategy

logger = logging.getLogger("trading_system.services.analysis_service")

class AnalysisService:
    """Service for portfolio and market analysis."""
    
    def __init__(self, config, data_service, portfolio_service, market_service, trading_service, cache_service):
        self.config = config
        self.data_service = data_service
        self.portfolio_service = portfolio_service
        self.market_service = market_service
        self.trading_service = trading_service
        self.cache_service = cache_service
    
    def analyze_portfolio(self, portfolio: Dict, account_balance: float, risk_profile: str = "medium",
                        extended_hours: bool = False, goals: Dict = None, quick_mode: bool = True) -> Dict[str, Any]:
        try:
            cache_key = f"portfolio_analysis:{hash(str(portfolio))}-{account_balance}-{risk_profile}-{extended_hours}"
            cached_result = self.cache_service.get(cache_key)
            if cached_result and quick_mode:
                return cached_result
            
            results = {}
            total_invested = 0.0
            total_current = 0.0
            
            # Get market sentiment for analysis
            market_sentiment = self.market_service.get_market_sentiment()
            
            # Calculate days remaining for recovery goals
            remaining_days = 0
            daily_goal = 0
            
            if goals:
                remaining_days = goals.get("days", 0)
                if goals.get("start_date"):
                    from datetime import datetime
                    start_date = datetime.strptime(goals["start_date"], "%Y-%m-%d")
                    days_passed = (datetime.now() - start_date).days
                    remaining_days = max(1, goals["days"] - days_passed)
                
                daily_goal = goals.get("target_recovery", 0) / max(1, remaining_days)
            
            # Get all tickers and prefetch prices
            tickers = list(portfolio.keys())
            price_map = self.data_service.get_current_prices(tickers)
            
            # Get trading parameters
            trading_params = self.trading_service.get_trading_parameters()
            
            # Create portfolio summary
            portfolio_summary = {
                "total_invested": total_invested,
                "valor_atual": total_current,
                "lucro_prejuizo": total_current - total_invested,
                "lucro_prejuizo_pct": ((total_current / total_invested - 1) * 100) if total_invested > 0 else 0.0,
                "saldo_disponivel": account_balance,
                "patrimonio_total": account_balance + total_current,
                "market_sentiment": market_sentiment,
                "meta_recuperacao": goals.get("target_recovery", 0) if goals else 0,
                "dias_restantes": remaining_days,
                "dias_totais": goals.get("days", 30) if goals else 30,
                "meta_diaria": daily_goal
            }
            
            # Final result
            final_result = {"ativos": results, "resumo": portfolio_summary}
            
            # Cache result
            self.cache_service.set(cache_key, final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return {"ativos": {}, "resumo": {}, "error": str(e)}
    
    def analyze_watchlist(self, watchlist: Dict, account_balance: float, risk_profile: str = "medium",
                        extended_hours: bool = False, goals: Dict = None, quick_mode: bool = True) -> Dict[str, Any]:
        try:
            return {}  # Simplified implementation
        except Exception as e:
            logger.error(f"Error analyzing watchlist: {e}")
            return {"error": str(e)}
    
    def generate_rebalance_plan(self, portfolio_analysis: Dict, watchlist_analysis: Dict, 
                               account_balance: float) -> Dict[str, Any]:
        try:
            return {"sell": [], "buy": [], "rebalance": [], "stats": {}}  # Simplified implementation
        except Exception as e:
            logger.error(f"Error generating rebalance plan: {e}")
            return {"sell": [], "buy": [], "rebalance": [], "stats": {}, "error": str(e)}
    
    def ensure_complete_analysis_results(self, portfolio_analysis: Dict) -> Dict:
        """Ensure that the analysis results have all required fields."""
        if "resumo" not in portfolio_analysis:
            portfolio_analysis["resumo"] = {}
        
        # Ensure required fields in resumo
        required_fields = [
            "total_invested", "valor_atual", "lucro_prejuizo", "lucro_prejuizo_pct",
            "saldo_disponivel", "patrimonio_total", "market_sentiment",
            "meta_recuperacao", "dias_restantes", "dias_totais", "meta_diaria"
        ]
        
        for field in required_fields:
            if field not in portfolio_analysis["resumo"]:
                portfolio_analysis["resumo"][field] = 0
        
        return portfolio_analysis