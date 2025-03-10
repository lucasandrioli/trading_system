import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger("trading_system.position_sizing")

class PositionSizing:
    """Cálculo do tamanho de posição."""
    
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
                                params: dict = None) -> Dict[str, Any]:
        """Calcula o tamanho ideal de posição considerando risco e capital disponível."""
        if params is None:
            params = {}

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
        commission = PositionSizing.calculate_xp_commission(allocation_value)
        
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
            total_cost = shares * price + PositionSizing.calculate_xp_commission(shares * price)
            if total_cost <= balance or shares == 1:
                break
            shares -= 1
        
        # Verificar se vale a pena (valor mínimo e comissão)
        if shares > 0:
            value = shares * price
            if value < 50.0:  # Valor mínimo para uma transação
                return 0
        
        return shares
        
    @staticmethod
    def calculate_xp_commission(value: float) -> float:
        """
        Calcula a comissão da XP com base no valor da operação.
        Nova estrutura de comissões: $1.00 para operações a partir de $50.
        """
        if value < 50:
            return 0.0
        else:
            return 1.00