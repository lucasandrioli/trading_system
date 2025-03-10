import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import csv
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class VisualizationService:
    """Service for generating visualizations and data exports."""
    
    @staticmethod
    def generate_portfolio_chart(history: List[Dict[str, Any]]) -> str:
        """Generate portfolio history chart as base64 encoded image."""
        try:
            if not history:
                return ""
            
            # Extract data
            dates = []
            total_values = []
            portfolio_values = []
            cash_values = []
            
            for entry in history:
                date = entry.get('date')
                total_value = entry.get('total_value', 0)
                portfolio_value = entry.get('portfolio_value', 0)
                cash_balance = entry.get('cash_balance', 0)
                
                dates.append(date)
                total_values.append(total_value)
                portfolio_values.append(portfolio_value)
                cash_values.append(cash_balance)
            
            # Create chart
            plt.figure(figsize=(12, 6))
            plt.plot(dates, total_values, 'b-', label='Total Value')
            plt.plot(dates, portfolio_values, 'g-', label='Portfolio Value')
            plt.plot(dates, cash_values, 'r-', label='Cash')
            
            plt.title('Portfolio History')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.grid(True)
            plt.legend()
            
            # If we have more than 10 dates, rotate labels
            if len(dates) > 10:
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save to memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error generating portfolio chart: {e}")
            return ""
    
    @staticmethod
    def generate_allocation_chart(portfolio: Dict[str, Any]) -> str:
        """Generate portfolio allocation chart as base64 encoded image."""
        try:
            if not portfolio or 'positions' not in portfolio:
                return ""
            
            # Extract data
            labels = []
            values = []
            
            for ticker, position in portfolio['positions'].items():
                value = position.get('quantity', 0) * position.get('current_price', 0)
                if value > 0:
                    labels.append(ticker)
                    values.append(value)
            
            # Add cash if provided
            if 'account_balance' in portfolio and portfolio['account_balance'] > 0:
                labels.append('Cash')
                values.append(portfolio['account_balance'])
            
            # Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Portfolio Allocation')
            
            # Save to memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error generating allocation chart: {e}")
            return ""
    
    @staticmethod
    def generate_technical_indicator_charts(df: pd.DataFrame, ticker: str) -> Dict[str, str]:
        """Generate technical indicator charts as base64 encoded images."""
        charts = {}
        
        try:
            if df.empty:
                return charts
            
            # Price chart with SMA and Bollinger Bands
            if 'Close' in df.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df['Close'], 'b-', label='Close')
                
                if 'SMA' in df.columns:
                    plt.plot(df.index, df['SMA'], 'g-', label='SMA')
                
                if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                    plt.plot(df.index, df['BB_upper'], 'r--', label='Upper BB')
                    plt.plot(df.index, df['BB_lower'], 'r--', label='Lower BB')
                
                plt.title(f'{ticker} Price Chart')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.grid(True)
                plt.legend()
                
                # Save to memory
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                
                # Convert to base64
                charts['price'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # RSI chart
            if 'RSI' in df.columns:
                plt.figure(figsize=(12, 4))
                plt.plot(df.index, df['RSI'], 'b-')
                plt.axhline(y=70, color='r', linestyle='-')
                plt.axhline(y=30, color='g', linestyle='-')
                plt.axhline(y=50, color='k', linestyle='--')
                
                plt.title(f'{ticker} RSI')
                plt.xlabel('Date')
                plt.ylabel('RSI')
                plt.grid(True)
                
                # Save to memory
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                
                # Convert to base64
                charts['rsi'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # MACD chart
            if all(x in df.columns for x in ['MACD_line', 'MACD_signal', 'MACD_hist']):
                plt.figure(figsize=(12, 4))
                plt.plot(df.index, df['MACD_line'], 'b-', label='MACD')
                plt.plot(df.index, df['MACD_signal'], 'r-', label='Signal')
                
                # Plot histogram as bars
                colors = ['g' if x > 0 else 'r' for x in df['MACD_hist']]
                plt.bar(df.index, df['MACD_hist'], color=colors, alpha=0.5)
                
                plt.title(f'{ticker} MACD')
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.grid(True)
                plt.legend()
                
                # Save to memory
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                
                # Convert to base64
                charts['macd'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating technical charts for {ticker}: {e}")
            return charts
    
    @staticmethod
    def export_portfolio_to_csv(portfolio: Dict[str, Any]) -> str:
        """Export portfolio to CSV format."""
        try:
            if not portfolio or 'positions' not in portfolio:
                return ""
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Ticker', 'Quantity', 'Average Price', 'Current Price', 
                           'Current Value', 'Profit/Loss', 'Profit/Loss %'])
            
            # Write positions
            for ticker, position in portfolio['positions'].items():
                quantity = position.get('quantity', 0)
                avg_price = position.get('avg_price', 0)
                current_price = position.get('current_price', 0)
                current_value = quantity * current_price
                profit_loss = current_value - (quantity * avg_price)
                profit_loss_pct = (current_price / avg_price - 1) * 100 if avg_price > 0 else 0
                
                writer.writerow([
                    ticker,
                    quantity,
                    f"${avg_price:.2f}",
                    f"${current_price:.2f}",
                    f"${current_value:.2f}",
                    f"${profit_loss:.2f}",
                    f"{profit_loss_pct:.2f}%"
                ])
            
            # Write summary
            writer.writerow([])
            writer.writerow(['Summary'])
            
            total_invested = sum(pos.get('quantity', 0) * pos.get('avg_price', 0) 
                               for pos in portfolio['positions'].values())
            total_value = sum(pos.get('quantity', 0) * pos.get('current_price', 0) 
                             for pos in portfolio['positions'].values())
            total_pnl = total_value - total_invested
            total_pnl_pct = (total_value / total_invested - 1) * 100 if total_invested > 0 else 0
            
            writer.writerow(['Total Invested', f"${total_invested:.2f}"])
            writer.writerow(['Current Value', f"${total_value:.2f}"])
            writer.writerow(['Cash Balance', f"${portfolio.get('account_balance', 0):.2f}"])
            writer.writerow(['Total Portfolio', f"${total_value + portfolio.get('account_balance', 0):.2f}"])
            writer.writerow(['Profit/Loss', f"${total_pnl:.2f}"])
            writer.writerow(['Profit/Loss %', f"{total_pnl_pct:.2f}%"])
            writer.writerow(['Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error exporting portfolio to CSV: {e}")
            return ""
    
    @staticmethod
    def export_history_to_csv(history: List[Dict[str, Any]]) -> str:
        """Export portfolio history to CSV format."""
        try:
            if not history:
                return ""
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Date', 'Portfolio Value', 'Cash Balance', 'Total Value'])
            
            # Write history entries
            for entry in history:
                date = entry.get('date', '')
                portfolio_value = entry.get('portfolio_value', 0)
                cash_balance = entry.get('cash_balance', 0)
                total_value = entry.get('total_value', 0)
                
                writer.writerow([
                    date,
                    f"${portfolio_value:.2f}",
                    f"${cash_balance:.2f}",
                    f"${total_value:.2f}"
                ])
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error exporting history to CSV: {e}")
            return ""