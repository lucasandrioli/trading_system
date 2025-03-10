import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from trading_system.core.analysis.technical_indicators import TechnicalIndicators
from trading_system.core.analysis.fundamental_analysis import FundamentalAnalysis
from trading_system.core.analysis.qualitative_analysis import QualitativeAnalysis
from trading_system.core.analysis.market_analysis import MarketAnalysis
from trading_system.core.strategy.strategy import Strategy
from trading_system.core.strategy.position_sizing import PositionSizing
from trading_system.utils.cache_manager import CacheManager
from trading_system.models.portfolio import Portfolio, Position, RecoveryGoal

logger = logging.getLogger(__name__)

class AnalysisService:
    """Service for portfolio and market analysis."""
    
    def __init__(self, data_dir: str, cache_manager: CacheManager, data_service=None, market_service=None):
        self.data_dir = data_dir
        self.cache_manager = cache_manager
        self.data_service = data_service
        self.market_service = market_service
    
    def analyze_ticker(self, ticker: str, risk_profile: str = "medium", 
                       days: int = 60, include_indicators: bool = True) -> Dict[str, Any]:
        """Analyze a single ticker with technical, fundamental and qualitative data."""
        try:
            # Check cache first
            cache_key = f"ticker_analysis:{ticker}:{risk_profile}:{days}:{include_indicators}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            # Get historical data
            if self.data_service:
                df = self.data_service.get_historical_data(ticker, days)
            else:
                from trading_system.core.data.data_loader import DataLoader
                df = DataLoader.get_asset_data(ticker, days)
            
            if df.empty:
                return {
                    "ticker": ticker,
                    "success": False, 
                    "message": "Insufficient data for analysis"
                }
            
            result = {
                "ticker": ticker,
                "success": True,
                "current_price": float(df['Close'].iloc[-1]),
                "date": df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
            }
            
            # Add technical indicators if requested
            if include_indicators:
                df_ind = TechnicalIndicators.add_all_indicators(df, {})
                tech_result = Strategy.technical_score(df_ind, risk_profile)
                result["technical"] = tech_result
            
            # Add fundamental analysis
            fund_result = FundamentalAnalysis.fundamental_score(ticker)
            result["fundamental"] = fund_result
            
            # Add qualitative analysis
            qual_result = QualitativeAnalysis.qualitative_score(ticker)
            result["qualitative"] = qual_result
            
            # Get market sentiment adjustment
            if self.market_service:
                market_sentiment = self.market_service.get_market_sentiment()
            else:
                market_sentiment = MarketAnalysis.get_market_sentiment()
            
            market_score = Strategy.market_adaptive_score(ticker, market_sentiment)
            result["market_score"] = market_score
            
            # Calculate combined score
            tech_score = tech_result.get("score", 0) if tech_result else 0
            fund_score = fund_result.get("fundamental_score", 0)
            qual_score = qual_result.get("qualitative_score", 0)
            
            total_score = (
                0.70 * tech_score +
                0.05 * fund_score +
                0.20 * qual_score +
                0.05 * market_score
            )
            
            result["total_score"] = float(np.clip(total_score, -100, 100))
            
            # Cache the result
            self.cache_manager.set(cache_key, result, 3600)  # 1 hour cache
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing ticker {ticker}: {e}")
            return {
                "ticker": ticker,
                "success": False,
                "message": str(e)
            }
    
    def analyze_portfolio(self, portfolio: Portfolio, risk_profile: str = "medium", 
                         extended_hours: bool = False) -> Dict[str, Any]:
        """Analyze a complete portfolio with recommendations."""
        try:
            # Check cache first
            cache_key = f"portfolio_analysis:{hash(str(portfolio))}-{risk_profile}-{extended_hours}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            results = {}
            total_invested = 0.0
            total_current = 0.0
            
            # Get market sentiment
            if self.market_service:
                market_sentiment = self.market_service.get_market_sentiment()
            else:
                market_sentiment = MarketAnalysis.get_market_sentiment()
            
            # Calculate days remaining for recovery goals
            remaining_days = 0
            daily_goal = 0
            
            if portfolio.goals:
                remaining_days = portfolio.goals.days_remaining
                daily_goal = portfolio.goals.daily_target
            
            # Get trailing stop data (empty initially, filled by decision engine)
            trailing_data = {}
            
            # Get all tickers and prefetch prices
            tickers = list(portfolio.positions.keys())
            
            if self.data_service:
                price_map = self.data_service.get_current_prices(tickers)
            else:
                from trading_system.core.data.data_loader import DataLoader
                price_map = DataLoader.get_realtime_prices_bulk(tickers)
            
            # Analyze each position
            for ticker, position in portfolio.positions.items():
                if self.data_service:
                    df = self.data_service.get_historical_data(ticker, 60, extended_hours=extended_hours)
                else:
                    from trading_system.core.data.data_loader import DataLoader
                    df = DataLoader.get_asset_data(ticker, 60, extended_hours=extended_hours)
                
                if df.empty:
                    results[ticker] = {"ticker": ticker, "error": "Insufficient historical data"}
                    continue
                
                # Update with current price if available
                current_price = price_map.get(ticker)
                if current_price and 'Close' in df.columns:
                    df.iloc[-1, df.columns.get_loc("Close")] = current_price
                
                # Calculate investment values
                quantity = position.quantity
                avg_price = position.avg_price
                invested_value = quantity * avg_price
                current_value = quantity * (current_price if current_price else df['Close'].iloc[-1])
                pnl = current_value - invested_value
                
                # Calculate daily gap for recovery goal
                daily_gap = max(0, daily_goal - pnl)
                
                # Generate trading decision
                decision = Strategy.decision_engine(
                    ticker, df, position.to_dict(), portfolio.account_balance, 
                    risk_profile, {}, daily_gap=daily_gap, daily_goal=daily_goal,
                    market_sentiment=market_sentiment, trailing_data=trailing_data,
                    goals=portfolio.goals.to_dict() if portfolio.goals else None,
                    remaining_days=remaining_days
                )
                
                # Calculate position sizing for buy decisions
                if decision["decision"].upper() in ["COMPRAR", "COMPRAR PARCIAL"]:
                    pos_size = PositionSizing.calculate_position_size(
                        ticker, df, portfolio.account_balance, risk_profile,
                        daily_gap=daily_gap, daily_goal=daily_goal
                    )
                    decision["position_sizing"] = pos_size
                
                # Add position value
                decision["position_value"] = current_value
                decision["quantity"] = quantity
                
                # Add to results
                results[ticker] = decision
                
                # Update totals if no error
                if "error" not in decision:
                    total_invested += invested_value
                    total_current += current_value
            
            # Create portfolio summary
            portfolio_summary = {
                "total_invested": total_invested,
                "valor_atual": total_current,
                "lucro_prejuizo": total_current - total_invested,
                "lucro_prejuizo_pct": ((total_current / total_invested - 1) * 100) if total_invested > 0 else 0.0,
                "saldo_disponivel": portfolio.account_balance,
                "patrimonio_total": portfolio.account_balance + total_current,
                "market_sentiment": market_sentiment,
                "meta_recuperacao": portfolio.goals.target_recovery if portfolio.goals else 0,
                "dias_restantes": remaining_days,
                "dias_totais": portfolio.goals.days if portfolio.goals else 30,
                "meta_diaria": daily_goal
            }
            
            # Final result
            final_result = {"ativos": results, "resumo": portfolio_summary}
            
            # Cache result
            self.cache_manager.set(cache_key, final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return {"ativos": {}, "resumo": {}, "error": str(e)}
    
    def analyze_watchlist(self, portfolio: Portfolio, risk_profile: str = "medium",
                         extended_hours: bool = False) -> Dict[str, Any]:
        """Analyze the watchlist and generate recommendations."""
        try:
            # Use cache when possible
            cache_key = f"watchlist_analysis:{hash(str(portfolio.watchlist))}-{risk_profile}-{extended_hours}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            results = {}
            
            # Get market sentiment
            if self.market_service:
                market_sentiment = self.market_service.get_market_sentiment()
            else:
                market_sentiment = MarketAnalysis.get_market_sentiment()
            
            # Calculate recovery goals data
            remaining_days = 0
            daily_goal = 0
            
            if portfolio.goals:
                remaining_days = portfolio.goals.days_remaining
                daily_goal = portfolio.goals.daily_target
            
            # Filter watchlist for monitored tickers
            monitored_tickers = [ticker for ticker, data in portfolio.watchlist.items() 
                                if data.get("monitor", False)]
            
            # Get prices in bulk
            if self.data_service:
                price_map = self.data_service.get_current_prices(monitored_tickers)
            else:
                from trading_system.core.data.data_loader import DataLoader
                price_map = DataLoader.get_realtime_prices_bulk(monitored_tickers)
            
            # Analyze each ticker
            for ticker in monitored_tickers:
                if self.data_service:
                    df = self.data_service.get_historical_data(ticker, 60, extended_hours=extended_hours)
                else:
                    from trading_system.core.data.data_loader import DataLoader
                    df = DataLoader.get_asset_data(ticker, 60, extended_hours=extended_hours)
                
                if df.empty:
                    results[ticker] = {"ticker": ticker, "error": "Insufficient data"}
                    continue
                
                # Update price if available
                current_price = price_map.get(ticker)
                if current_price and 'Close' in df.columns:
                    df.iloc[-1, df.columns.get_loc("Close")] = current_price
                
                # Create empty position
                fake_position = {"quantity": 0, "avg_price": 0}
                
                # Set daily gap for recovery
                daily_gap = daily_goal if daily_goal > 0 else 0
                
                # Generate trading decision
                decision = Strategy.decision_engine(
                    ticker, df, fake_position, portfolio.account_balance, risk_profile, {},
                    daily_gap=daily_gap, daily_goal=daily_goal, market_sentiment=market_sentiment,
                    goals=portfolio.goals.to_dict() if portfolio.goals else None,
                    remaining_days=remaining_days
                )
                
                # Calculate position sizing for buy decisions
                if decision.get("decision") in ["COMPRAR", "COMPRAR PARCIAL"]:
                    pos_size = PositionSizing.calculate_position_size(
                        ticker, df, portfolio.account_balance, risk_profile,
                        daily_gap=daily_gap, daily_goal=daily_goal
                    )
                    decision["position_sizing"] = pos_size
                
                results[ticker] = decision
            
            # Cache the results
            self.cache_manager.set(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing watchlist: {e}")
            return {"error": str(e)}
    
    def generate_rebalance_plan(self, portfolio_analysis: Dict, watchlist_analysis: Dict, 
                               portfolio: Portfolio) -> Dict[str, Any]:
        """Generate a portfolio rebalancing plan."""
        try:
            # Use cache when possible
            cache_key = f"rebalance_plan:{hash(str(portfolio_analysis))}-{hash(str(watchlist_analysis))}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            plan = {"sell": [], "buy": [], "rebalance": [], "stats": {}}
            
            # Counters
            sell_capital = 0.0
            current_positions = {}
            original_value = 0.0
            
            # Simplified implementation: direct from portfolio_service code
            # 1. Identify assets to sell
            for ticker, details in portfolio_analysis.get("ativos", {}).items():
                decision = details.get("decision", "").upper()
                qty = details.get("quantity", 0)
                price = details.get("current_price", 0)
                
                # Register current positions
                if qty > 0:
                    current_positions[ticker] = {"quantity": qty, "price": price}
                    original_value += qty * price
                
                # Process sell decisions
                if decision in ["VENDER", "REALIZAR LUCRO PARCIAL", "REDUZIR"] and qty > 0:
                    sell_qty = qty if decision == "VENDER" else max(1, int(qty * 0.5))
                    operation_value = sell_qty * price
                    
                    # Skip very small transactions
                    if operation_value < 50.0:
                        logger.info(f"Ignoring small sale of {ticker}: ${operation_value:.2f}")
                        continue
                    
                    # Calculate commission
                    commission = self._calculate_commission(operation_value)
                    capital_freed = operation_value - commission
                    sell_capital += capital_freed
                    
                    # Add to sell plan
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
            
            # 2. Identify buy candidates (simplified for watchlist items)
            buy_candidates = []
            for ticker, details in watchlist_analysis.items():
                decision = details.get("decision", "").upper()
                if decision in ["COMPRAR", "COMPRAR PARCIAL"]:
                    score = details.get("total_score", 0)
                    pos_size = details.get("position_sizing", {})
                    suggested_qty = pos_size.get("suggested_shares", 0)
                    price = details.get("current_price", 0)
                    
                    # Skip invalid candidates
                    if suggested_qty < 1 or price <= 0 or score < 40:
                        continue
                    
                    # Add to buy candidates
                    buy_candidates.append({
                        "ticker": ticker,
                        "score": score,
                        "suggested_qty": suggested_qty,
                        "current_price": price,
                        "reason": details.get("justificativa", ""),
                        "operation_value": suggested_qty * price
                    })
            
            # Sort by score and add top candidates to the buy plan
            buy_candidates.sort(key=lambda x: x["score"], reverse=True)
            available_capital = portfolio.account_balance + sell_capital
            
            for candidate in buy_candidates[:5]:  # Limit to top 5
                qty = candidate["suggested_qty"]
                price = candidate["current_price"]
                
                operation_value = qty * price
                commission = self._calculate_commission(operation_value)
                total_cost = operation_value + commission
                
                # Adjust quantity if insufficient funds
                if total_cost > available_capital * 0.95:
                    adjusted_qty = int((available_capital * 0.95 - commission) / price)
                    
                    if adjusted_qty >= 1:
                        operation_value = adjusted_qty * price
                        commission = self._calculate_commission(operation_value)
                        total_cost = operation_value + commission
                        qty = adjusted_qty
                    else:
                        continue
                
                # Skip small transactions
                if operation_value < 50.0:
                    continue
                
                # Add to buy plan
                plan["buy"].append({
                    "ticker": candidate["ticker"],
                    "buy_quantity": qty,
                    "current_price": price,
                    "operation_value": operation_value,
                    "commission": commission,
                    "total_cost": total_cost,
                    "reason": candidate["reason"]
                })
                
                available_capital -= total_cost
                
                # Stop if insufficient funds
                if available_capital < 100:
                    break
            
            # Calculate statistics for the plan
            total_commission = sum(item["commission"] for item in plan["sell"]) + sum(item["commission"] for item in plan["buy"])
            total_buy_cost = sum(item["total_cost"] for item in plan["buy"])
            
            plan["stats"] = {
                "saldo_inicial": portfolio.account_balance,
                "capital_liberado_vendas": sell_capital,
                "capital_total_disponivel": portfolio.account_balance + sell_capital,
                "capital_usado_compras": total_buy_cost,
                "saldo_remanescente": portfolio.account_balance + sell_capital - total_buy_cost,
                "total_comissoes": total_commission,
                "numero_vendas": len(plan["sell"]),
                "numero_rebalanceamentos": len(plan["rebalance"]),
                "numero_compras": len(plan["buy"])
            }
            
            # Cache result
            self.cache_manager.set(cache_key, plan)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating rebalance plan: {e}")
            return {"sell": [], "buy": [], "rebalance": [], "stats": {}, "error": str(e)}
    
    def _calculate_commission(self, value: float) -> float:
        """Calculate commission for a trade."""
        if value < 50:
            return 0.0
        else:
            return 1.00  # Flat commission for simplicity