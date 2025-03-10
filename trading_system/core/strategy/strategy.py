import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional


from trading_system.core.analysis.technical_indicators import TechnicalIndicators
from trading_system.core.strategy.position_sizing import PositionSizing

# Configure logging
logger = logging.getLogger("trading_system.strategy")

class Strategy:
    """Trading strategy implementation."""
    
    @staticmethod
    def technical_score(df: pd.DataFrame, risk_profile: str = "medium", params: dict = None) -> Dict[str, Any]:
        """Calcula o score técnico com base nos indicadores do DataFrame."""
        if params is None:
            params = {}
            
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
        risk_mult = {"low": 0.8, "medium": 1.0, "high": 1.5, "ultra": 2.0}.get(risk_profile, 1.0)
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
            import yfinance as yf
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
        from trading_system.core.analysis.fundamental_analysis import FundamentalAnalysis
        from trading_system.core.analysis.qualitative_analysis import QualitativeAnalysis
        
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
            params.get("weight_tech", 0.70) * technical_score +
            params.get("weight_qual", 0.20) * qualitative_score +
            params.get("weight_fund", 0.05) * fundamental_score +
            params.get("weight_market", 0.05) * market_score
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
        buy_threshold = params.get("decision_buy_threshold", 60) * (1 - adj_factor * gap_ratio)
        buy_partial_threshold = params.get("decision_buy_partial_threshold", 20) * (1 - adj_factor * gap_ratio)
        sell_threshold = params.get("decision_sell_threshold", -60) * (1 + adj_factor * gap_ratio)
        sell_partial_threshold = params.get("decision_sell_partial_threshold", -20) * (1 + adj_factor * gap_ratio)
        
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
            result_ts = Strategy.check_trailing_stop(ticker, position, current_price, trailing_data, params)
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
        
    @staticmethod
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