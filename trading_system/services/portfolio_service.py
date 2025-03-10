import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from core.analysis.fundamental_analysis import FundamentalAnalysis
from core.analysis.technical_indicators import TechnicalIndicators
from core.analysis.market_analysis import MarketAnalysis
from core.analysis.qualitative_analysis import QualitativeAnalysis
from core.strategy.strategy import Strategy
from core.strategy.position_sizing import PositionSizing
from data.data_loader import DataLoader
from utils.cache_manager import CacheManager
from models.portfolio import Portfolio, Position, TradeHistory, RecoveryGoal

logger = logging.getLogger(__name__)

class PortfolioService:
    """Service for managing portfolio operations and analysis."""
    
    def __init__(self, data_dir: str, cache_manager: CacheManager):
        self.data_dir = data_dir
        self.portfolio_file = os.path.join(data_dir, 'portfolio.json')
        self.history_file = os.path.join(data_dir, 'portfolio_history.json')
        self.trailing_file = os.path.join(data_dir, 'trailing_data.json')
        self.cache_manager = cache_manager
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_portfolio(self) -> Portfolio:
        """Load portfolio data from file."""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                    
                    # Build portfolio object from JSON data
                    portfolio = Portfolio(
                        positions={ticker: Position(**details) for ticker, details in data.get('portfolio', {}).items()},
                        watchlist=data.get('watchlist', {}),
                        account_balance=data.get('account_balance', 0.0)
                    )
                    
                    # Add recovery goals if present
                    if 'goals' in data:
                        portfolio.goals = RecoveryGoal(**data['goals'])
                    
                    return portfolio
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
        
        # Return empty portfolio if loading fails
        return Portfolio(positions={}, watchlist={}, account_balance=0.0)
    
    def save_portfolio(self, portfolio: Portfolio) -> bool:
        """Save portfolio data to file."""
        try:
            # Convert portfolio object to serializable dict
            data = {
                'portfolio': {ticker: position.to_dict() for ticker, position in portfolio.positions.items()},
                'watchlist': portfolio.watchlist,
                'account_balance': portfolio.account_balance
            }
            
            # Add goals if present
            if portfolio.goals:
                data['goals'] = portfolio.goals.to_dict()
            
            with open(self.portfolio_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
            return False
    
    def load_portfolio_history(self) -> List[Dict[str, Any]]:
        """Load portfolio history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio history: {e}")
        return []
    
    def save_portfolio_history(self, history: List[Dict[str, Any]]) -> bool:
        """Save portfolio history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving portfolio history: {e}")
            return False
    
    def load_trailing_data(self) -> Dict[str, Any]:
        """Load trailing stop data from file."""
        try:
            if os.path.exists(self.trailing_file):
                with open(self.trailing_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading trailing data: {e}")
        return {}
    
    def save_trailing_data(self, data: Dict[str, Any]) -> bool:
        """Save trailing stop data to file."""
        try:
            with open(self.trailing_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving trailing data: {e}")
            return False
    
    def add_portfolio_snapshot(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Add a snapshot of current portfolio state to history."""
        total_value = sum(position.quantity * position.current_price for position in portfolio.positions.values())
        
        snapshot = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": total_value,
            "cash_balance": portfolio.account_balance,
            "total_value": total_value + portfolio.account_balance,
            "positions": {
                ticker: {
                    "quantity": position.quantity,
                    "price": position.current_price,
                    "value": position.quantity * position.current_price
                } for ticker, position in portfolio.positions.items()
            }
        }
        
        # Add to history
        history = self.load_portfolio_history()
        history.append(snapshot)
        
        # Keep last 90 days only
        if len(history) > 90:
            history = history[-90:]
        
        self.save_portfolio_history(history)
        return snapshot
    
    def analyze_portfolio(self, portfolio: Portfolio, risk_profile: str = "medium", 
                          extended_hours: bool = False) -> Dict[str, Any]:
        """Analyze the portfolio and generate recommendations."""
        # Implements the core portfolio analysis logic
        try:
            # Use cache when possible
            cache_key = f"portfolio_analysis:{hash(str(portfolio))}-{risk_profile}-{extended_hours}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            trailing_data = self.load_trailing_data()
            results = {}
            total_invested = 0.0
            total_current = 0.0
            
            # Get market sentiment for analysis
            market_sentiment = MarketAnalysis.get_market_sentiment()
            
            # Calculate days remaining for recovery goals
            remaining_days = 0
            daily_goal = 0
            
            if portfolio.goals:
                remaining_days = portfolio.goals.days
                if portfolio.goals.start_date:
                    start_date = datetime.strptime(portfolio.goals.start_date, "%Y-%m-%d")
                    days_passed = (datetime.now() - start_date).days
                    remaining_days = max(1, portfolio.goals.days - days_passed)
                
                daily_goal = portfolio.goals.target_recovery / max(1, remaining_days)
            
            # Get all tickers and prefetch prices
            tickers = list(portfolio.positions.keys())
            price_map = DataLoader.get_realtime_prices_bulk(tickers)
            
            # Analyze each position
            for ticker, position in portfolio.positions.items():
                df = DataLoader.get_asset_data(ticker, days=60, extended_hours=extended_hours)
                
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
                    risk_profile, daily_gap=daily_gap, daily_goal=daily_goal,
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
            
            # Save updated trailing data
            self.save_trailing_data(trailing_data)
            
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
            market_sentiment = MarketAnalysis.get_market_sentiment()
            
            # Calculate recovery goals data
            remaining_days = 0
            daily_goal = 0
            
            if portfolio.goals:
                remaining_days = portfolio.goals.days
                if portfolio.goals.start_date:
                    start_date = datetime.strptime(portfolio.goals.start_date, "%Y-%m-%d")
                    days_passed = (datetime.now() - start_date).days
                    remaining_days = max(1, portfolio.goals.days - days_passed)
                
                daily_goal = portfolio.goals.target_recovery / max(1, remaining_days)
            
            # Filter watchlist for monitored tickers
            monitored_tickers = [ticker for ticker, data in portfolio.watchlist.items() 
                                if data.get("monitor", False)]
            
            # Get prices in bulk
            price_map = DataLoader.get_realtime_prices_bulk(monitored_tickers)
            
            # Analyze each ticker
            for ticker in monitored_tickers:
                df = DataLoader.get_asset_data(ticker, days=60, extended_hours=extended_hours)
                
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
                    ticker, df, fake_position, portfolio.account_balance, risk_profile,
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
            
            # 2. Identify buy candidates
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
            
            # 3. Identify rebalance candidates
            rebalance_candidates = []
            for ticker, details in portfolio_analysis.get("ativos", {}).items():
                score = details.get("total_score", 0)
                qty = details.get("quantity", 0)
                price = details.get("current_price", 0)
                position_profit_pct = details.get("position_profit_pct", 0)
                
                # Check for underperforming assets
                if (score < -20 or position_profit_pct < -3.0) and qty > 0:
                    already_in_sell = any(s['ticker'] == ticker for s in plan["sell"])
                    
                    if not already_in_sell:
                        # Calculate quantity for rebalancing
                        sell_qty = int(qty * 0.5)
                        if sell_qty < 1:
                            continue
                        
                        operation_value = sell_qty * price
                        if operation_value < 50.0:
                            continue
                        
                        commission = self._calculate_commission(operation_value)
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
            
            # Sort candidates
            rebalance_candidates.sort(key=lambda x: x["score"])
            buy_candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # 4. Check if additional rebalancing is needed
            additional_capital_needed = 0
            for buy in buy_candidates[:3]:
                additional_capital_needed += buy["operation_value"] + self._calculate_commission(buy["operation_value"])
            
            available_capital = portfolio.account_balance + sell_capital
            capital_gap = max(0, additional_capital_needed - available_capital)
            
            # 5. Add rebalancing if needed
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
            
            # 6. Distribute available capital for buys
            available_capital = portfolio.account_balance + sell_capital
            total_commission = 0
            
            for candidate in buy_candidates:
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
                total_commission += commission
                
                # Limit number of buys
                if len(plan["buy"]) >= 5 or available_capital < 100:
                    break
            
            # 7. Calculate plan statistics
            plan["stats"] = {
                "saldo_inicial": portfolio.account_balance,
                "capital_liberado_vendas": sell_capital,
                "capital_total_disponivel": portfolio.account_balance + sell_capital,
                "capital_usado_compras": sum(item["total_cost"] for item in plan["buy"]),
                "saldo_remanescente": portfolio.account_balance + sell_capital - sum(item["total_cost"] for item in plan["buy"]),
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
    
    def execute_rebalance_plan(self, portfolio: Portfolio, rebalance_plan: Dict[str, Any]) -> Tuple[Portfolio, Dict[str, Any]]:
        """Execute a rebalance plan, returning updated portfolio and results."""
        results = {
            "sells_executed": [],
            "buys_executed": [],
            "rebalances_executed": [],
            "errors": [],
            "starting_balance": portfolio.account_balance,
            "ending_balance": portfolio.account_balance,
            "capital_freed": 0,
            "capital_invested": 0,
            "total_commission": 0,
            "executed_transactions": 0
        }
        
        # Make a copy of the portfolio
        updated_portfolio = Portfolio(
            positions={k: Position(**v.to_dict()) for k, v in portfolio.positions.items()},
            watchlist=portfolio.watchlist.copy(),
            account_balance=portfolio.account_balance,
            goals=portfolio.goals.copy() if portfolio.goals else None
        )
        
        # Execute sells
        for sell in rebalance_plan.get("sell", []):
            try:
                ticker = sell["ticker"]
                qty = sell["sell_quantity"]
                price = sell["current_price"]
                
                if ticker not in updated_portfolio.positions:
                    results["errors"].append({
                        "ticker": ticker,
                        "action": "SELL",
                        "success": False,
                        "error": "Asset not in portfolio"
                    })
                    continue
                
                position = updated_portfolio.positions[ticker]
                if position.quantity < qty:
                    results["errors"].append({
                        "ticker": ticker,
                        "action": "SELL",
                        "success": False,
                        "error": f"Insufficient quantity. Have: {position.quantity}, Need: {qty}"
                    })
                    continue
                
                # Calculate values
                operation_value = qty * price
                commission = self._calculate_commission(operation_value)
                net_value = operation_value - commission
                
                # Update position
                position.quantity -= qty
                position.last_sell = {
                    "price": price,
                    "quantity": qty,
                    "date": datetime.now().isoformat()
                }
                
                # Remove if quantity is 0
                if position.quantity <= 0:
                    del updated_portfolio.positions[ticker]
                
                # Update balance
                updated_portfolio.account_balance += net_value
                
                # Update results
                results["capital_freed"] += net_value
                results["total_commission"] += commission
                results["sells_executed"].append({
                    "ticker": ticker,
                    "quantity": qty,
                    "price": price,
                    "gross_value": operation_value,
                    "commission": commission,
                    "net_value": net_value,
                    "remaining_quantity": position.quantity if ticker in updated_portfolio.positions else 0
                })
                
            except Exception as e:
                logger.error(f"Error executing sell for {sell.get('ticker', 'unknown')}: {e}")
                results["errors"].append({
                    "ticker": sell.get("ticker", "unknown"),
                    "action": "SELL",
                    "success": False,
                    "error": str(e)
                })
        
        # Execute rebalancing sells
        for rebalance in rebalance_plan.get("rebalance", []):
            try:
                ticker = rebalance["ticker"]
                qty = rebalance["sell_quantity"]
                price = rebalance["current_price"]
                
                if ticker not in updated_portfolio.positions:
                    results["errors"].append({
                        "ticker": ticker,
                        "action": "REBALANCE",
                        "success": False,
                        "error": "Asset not in portfolio"
                    })
                    continue
                
                position = updated_portfolio.positions[ticker]
                if position.quantity < qty:
                    results["errors"].append({
                        "ticker": ticker,
                        "action": "REBALANCE",
                        "success": False,
                        "error": f"Insufficient quantity. Have: {position.quantity}, Need: {qty}"
                    })
                    continue
                
                # Calculate values
                operation_value = qty * price
                commission = self._calculate_commission(operation_value)
                net_value = operation_value - commission
                
                # Update position
                position.quantity -= qty
                position.last_sell = {
                    "price": price,
                    "quantity": qty,
                    "date": datetime.now().isoformat()
                }
                
                # Remove if quantity is 0
                if position.quantity <= 0:
                    del updated_portfolio.positions[ticker]
                
                # Update balance
                updated_portfolio.account_balance += net_value
                
                # Update results
                results["capital_freed"] += net_value
                results["total_commission"] += commission
                results["rebalances_executed"].append({
                    "ticker": ticker,
                    "quantity": qty,
                    "price": price,
                    "gross_value": operation_value,
                    "commission": commission,
                    "net_value": net_value,
                    "remaining_quantity": position.quantity if ticker in updated_portfolio.positions else 0
                })
                
            except Exception as e:
                logger.error(f"Error executing rebalance for {rebalance.get('ticker', 'unknown')}: {e}")
                results["errors"].append({
                    "ticker": rebalance.get("ticker", "unknown"),
                    "action": "REBALANCE",
                    "success": False,
                    "error": str(e)
                })
        
        # Set intermediate balance for reporting
        results["balance_after_sells"] = updated_portfolio.account_balance
        
        # Execute buys
        for buy in rebalance_plan.get("buy", []):
            try:
                ticker = buy["ticker"]
                qty = buy["buy_quantity"]
                price = buy["current_price"]
                
                # Calculate values
                operation_value = qty * price
                commission = self._calculate_commission(operation_value)
                total_cost = operation_value + commission
                
                # Check if balance is sufficient
                if total_cost > updated_portfolio.account_balance:
                    # Try to adjust quantity
                    adjusted_qty = self._max_affordable_quantity(price, updated_portfolio.account_balance)
                    
                    if adjusted_qty < 1:
                        results["errors"].append({
                            "ticker": ticker,
                            "action": "BUY",
                            "success": False,
                            "error": "Insufficient balance"
                        })
                        continue
                    
                    # Recalculate with adjusted quantity
                    qty = adjusted_qty
                    operation_value = qty * price
                    commission = self._calculate_commission(operation_value)
                    total_cost = operation_value + commission
                
                # Check if position already exists
                if ticker in updated_portfolio.positions:
                    position = updated_portfolio.positions[ticker]
                    old_qty = position.quantity
                    old_price = position.avg_price
                    
                    # Calculate new average price
                    new_qty = old_qty + qty
                    new_avg_price = ((old_qty * old_price) + (qty * price)) / new_qty
                    
                    # Update position
                    position.quantity = new_qty
                    position.avg_price = new_avg_price
                    position.current_price = price
                    position.last_buy = {
                        "price": price,
                        "quantity": qty,
                        "date": datetime.now().isoformat()
                    }
                else:
                    # Create new position
                    updated_portfolio.positions[ticker] = Position(
                        symbol=ticker,
                        quantity=qty,
                        avg_price=price,
                        current_price=price,
                        last_buy={
                            "price": price,
                            "quantity": qty,
                            "date": datetime.now().isoformat()
                        }
                    )
                
                # Update balance
                updated_portfolio.account_balance -= total_cost
                
                # Update results
                results["capital_invested"] += total_cost
                results["total_commission"] += commission
                results["buys_executed"].append({
                    "ticker": ticker,
                    "quantity": qty,
                    "price": price,
                    "gross_value": operation_value,
                    "commission": commission,
                    "total_cost": total_cost,
                    "new_quantity": updated_portfolio.positions[ticker].quantity,
                    "new_avg_price": updated_portfolio.positions[ticker].avg_price
                })
                
            except Exception as e:
                logger.error(f"Error executing buy for {buy.get('ticker', 'unknown')}: {e}")
                results["errors"].append({
                    "ticker": buy.get("ticker", "unknown"),
                    "action": "BUY",
                    "success": False,
                    "error": str(e)
                })
        
        # Final results
        results["ending_balance"] = updated_portfolio.account_balance
        results["executed_transactions"] = len(results["sells_executed"]) + len(results["rebalances_executed"]) + len(results["buys_executed"])
        results["timestamp"] = datetime.now().isoformat()
        
        # Add portfolio summary
        results["updated_portfolio"] = self._calculate_portfolio_summary(updated_portfolio)
        
        return updated_portfolio, results
    
    def _calculate_commission(self, value: float) -> float:
        """Calculate commission based on transaction value."""
        if value < 50:
            return 0.0
        else:
            return 1.00
    
    def _max_affordable_quantity(self, price: float, available_balance: float) -> int:
        """Calculate the maximum affordable quantity of shares."""
        if price <= 0:
            return 0
        
        # Initial guess
        qty = int(available_balance / price)
        
        # Adjust for commission
        while qty > 0:
            cost = qty * price
            commission = self._calculate_commission(cost)
            total = cost + commission
            
            if total <= available_balance:
                return qty
            
            qty -= 1
        
        return 0
    
    def _calculate_portfolio_summary(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Calculate summary statistics for a portfolio."""
        total_invested = 0
        total_current_value = 0
        positions = []
        
        # Get current prices
        tickers = list(portfolio.positions.keys())
        current_prices = DataLoader.get_realtime_prices_bulk(tickers)
        
        for ticker, position in portfolio.positions.items():
            invested_value = position.quantity * position.avg_price
            
            current_price = current_prices.get(ticker, position.avg_price)
            current_value = position.quantity * current_price
            
            profit_loss = current_value - invested_value
            profit_loss_pct = (profit_loss / invested_value * 100) if invested_value > 0 else 0
            
            positions.append({
                "ticker": ticker,
                "quantity": position.quantity,
                "avg_price": position.avg_price,
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
            "cash_balance": portfolio.account_balance,
            "total_portfolio_value": total_current_value + portfolio.account_balance,
            "position_count": len(positions),
            "timestamp": datetime.now().isoformat()
        }