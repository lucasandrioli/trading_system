from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_wtf.csrf import CSRFProtect
from io import BytesIO
import json
import base64
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Importações absolutas
from trading_system.core.analysis.technical_indicators import TechnicalIndicators
from trading_system.core.analysis.fundamental_analysis import FundamentalAnalysis
from trading_system.core.analysis.qualitative_analysis import QualitativeAnalysis
from trading_system.core.analysis.market_analysis import MarketAnalysis
from trading_system.core.strategy.strategy import Strategy
from trading_system.core.strategy.position_sizing import PositionSizing
from trading_system.core.data.data_loader import DataLoader
from trading_system.services.portfolio_service import PortfolioService
from trading_system.services.market_service import MarketService
from trading_system.services.form_service import FormService
from trading_system.services.trading_service import TradingService
from trading_system.services.system_service import SystemService
from trading_system.services.history_service import HistoryService
from trading_system.utils.cache_manager import CacheManager
from trading_system.models.portfolio import Portfolio, Position, TradeHistory, RecoveryGoal

# Create blueprint
web_bp = Blueprint('web', __name__)

# Services
portfolio_service = None
market_service = None
form_service = None
cache_manager = None

def init_services(data_dir, cache_file):
    """Initialize services needed by routes."""
    global portfolio_service, market_service, form_service, cache_manager
    
    cache_manager = CacheManager(cache_file)
    portfolio_service = PortfolioService(data_dir, cache_manager)
    market_service = MarketService(cache_manager)
    form_service = FormService()

# Dashboard route
@web_bp.route('/')
def index():
    """Dashboard/home page."""
    portfolio = portfolio_service.load_portfolio()
    
    # Calculate portfolio totals
    total_invested = portfolio.total_invested
    portfolio_value = portfolio.portfolio_value
    account_balance = portfolio.account_balance
    total_value = portfolio.total_value
    profit_loss = portfolio.profit_loss
    profit_loss_pct = portfolio.profit_loss_pct
    
    # Get market sentiment 
    market_sentiment = market_service.get_market_sentiment()
    
    # Get recovery tracker status
    recovery_status = {}
    if portfolio.goals and portfolio.goals.target_recovery > 0:
        history = portfolio_service.load_portfolio_history()
        if history:
            # Calculate recovery metrics
            first_value = history[0]['total_value'] if history else total_value
            current_pnl = total_value - first_value
            progress_pct = (current_pnl / portfolio.goals.target_recovery) * 100 if portfolio.goals.target_recovery else 0
            
            # Get days info
            days_passed = portfolio.goals.days_passed
            days_remaining = portfolio.goals.days_remaining
            days_total = portfolio.goals.days
            
            # On track status
            on_track = progress_pct >= (days_passed / max(1, days_total) * 100) if days_total > 0 else False
            
            recovery_status = {
                'current_pnl': current_pnl,
                'progress_pct': progress_pct,
                'days_passed': days_passed,
                'days_remaining': days_remaining,
                'daily_target': portfolio.goals.daily_target,
                'on_track': on_track
            }
    
    # Prepare summary data
    summary = {
        'total_invested': total_invested,
        'portfolio_value': portfolio_value,
        'account_balance': account_balance,
        'total_value': total_value,
        'profit_loss': profit_loss,
        'profit_loss_pct': profit_loss_pct,
        'position_count': len(portfolio.positions),
        'market_sentiment': market_sentiment,
        'recovery_status': recovery_status
    }
    
    # Take a daily snapshot if needed
    history = portfolio_service.load_portfolio_history()
    if not history or history[-1]['date'] != datetime.now().strftime("%Y-%m-%d"):
        portfolio_service.add_portfolio_snapshot(portfolio)
    
    # Check for updates
    update_info = {"current_version": "2.0.0", "update_available": False}
    
    return render_template(
        'index.html', 
        portfolio=portfolio.positions, 
        watchlist=portfolio.watchlist,
        summary=summary,
        goals=portfolio.goals,
        update_info=update_info,
    )

@web_bp.route('/portfolio')
def portfolio():
    """Portfolio management page."""
    portfolio_obj = portfolio_service.load_portfolio()
    form = form_service.create_asset_form()
    return render_template('portfolio.html', portfolio=portfolio_obj.positions, form=form)

@web_bp.route('/portfolio/add', methods=['POST'])
def add_asset():
    """Add asset to portfolio."""
    form = form_service.create_asset_form()
    
    if form.validate_on_submit():
        portfolio = portfolio_service.load_portfolio()
        ticker = form.ticker.data.upper()
        
        # Validate ticker
        ticker_valid = market_service.validate_ticker(ticker)
        
        if ticker_valid:
            # Get current price
            df = market_service.get_ticker_data(ticker, days=1)
            current_price = float(df[0]['Close']) if df else form.avg_price.data
            
            # Check if ticker already exists in portfolio
            is_update = ticker in portfolio.positions
            
            if is_update:
                # Update existing position
                position = portfolio.positions[ticker]
                old_qty = position.quantity
                old_price = position.avg_price
                new_qty = form.quantity.data
                
                # Calculate additional investment
                additional_investment = 0
                if new_qty > old_qty:
                    additional_qty = new_qty - old_qty
                    additional_investment = additional_qty * form.avg_price.data
                    
                    # Check if sufficient balance
                    if additional_investment > portfolio.account_balance:
                        flash(f'Saldo insuficiente para adicionar {additional_qty} ações. Necessário: ${additional_investment:.2f}, Disponível: ${portfolio.account_balance:.2f}', 'warning')
                
                # Calculate new average price
                total_qty = new_qty
                if total_qty > 0:
                    avg_price = ((old_qty * old_price) + ((new_qty - old_qty) * form.avg_price.data)) / total_qty
                else:
                    avg_price = 0
                
                # Adjust account balance
                if new_qty < old_qty:
                    returned_funds = (old_qty - new_qty) * old_price
                    portfolio.account_balance += returned_funds
                elif new_qty > old_qty:
                    portfolio.account_balance -= additional_investment
                
                # Update position
                position.quantity = total_qty
                position.avg_price = avg_price
                position.current_price = current_price
                position.current_position = total_qty * current_price
                position.last_buy = {
                    'price': form.avg_price.data,
                    'quantity': form.quantity.data,
                    'date': datetime.now().isoformat()
                }
                
                flash(f'Ativo {ticker} atualizado com sucesso!', 'success')
                
            else:
                # Add new position
                investment_amount = form.quantity.data * form.avg_price.data
                
                # Check if sufficient balance
                if investment_amount > portfolio.account_balance:
                    flash(f'Saldo insuficiente para comprar {form.quantity.data} ações. Necessário: ${investment_amount:.2f}, Disponível: ${portfolio.account_balance:.2f}', 'danger')
                    return redirect(url_for('web.portfolio'))
                
                # Deduct from account balance
                portfolio.account_balance -= investment_amount
                
                # Create new position
                portfolio.positions[ticker] = Position(
                    symbol=ticker,
                    quantity=form.quantity.data,
                    avg_price=form.avg_price.data,
                    current_price=current_price,
                    current_position=form.quantity.data * current_price,
                    last_buy={
                        'price': form.avg_price.data,
                        'quantity': form.quantity.data,
                        'date': datetime.now().isoformat()
                    }
                )
                
                flash(f'Ativo {ticker} adicionado com sucesso!', 'success')
            
            # Save portfolio
            portfolio_service.save_portfolio(portfolio)
            
            # Clear cache for this asset
            cache_manager.delete(f"asset_analysis:{ticker}")
            cache_manager.delete(f"portfolio_analysis:{hash(str(portfolio))}")
            
            # Take a snapshot
            portfolio_service.add_portfolio_snapshot(portfolio)
            
        else:
            flash(f'Ticker {ticker} inválido ou não encontrado!', 'danger')
    
    return redirect(url_for('web.portfolio'))

@web_bp.route('/portfolio/edit/<ticker>', methods=['GET', 'POST'])
def edit_asset(ticker):
    """Edit a portfolio asset."""
    portfolio = portfolio_service.load_portfolio()
    
    if ticker not in portfolio.positions:
        flash(f'Ativo {ticker} não encontrado na carteira!', 'danger')
        return redirect(url_for('web.portfolio'))
    
    position = portfolio.positions[ticker]
    
    if request.method == 'POST':
        form = form_service.create_asset_form()
        if form.validate_on_submit():
            # Update position
            new_qty = form.quantity.data
            new_price = form.avg_price.data
            
            # Get current price
            current_price = market_service.get_current_prices([ticker]).get(ticker, new_price)
            
            # Update position
            position.quantity = new_qty
            position.avg_price = new_price
            position.current_price = current_price
            position.current_position = new_qty * current_price
            
            # Save portfolio
            portfolio_service.save_portfolio(portfolio)
            
            # Clear cache
            cache_manager.delete(f"asset_analysis:{ticker}")
            cache_manager.delete(f"portfolio_analysis:{hash(str(portfolio))}")
            
            # Take snapshot
            portfolio_service.add_portfolio_snapshot(portfolio)
            
            flash(f'Ativo {ticker} atualizado com sucesso!', 'success')
            return redirect(url_for('web.portfolio'))
    else:
        # Create form with existing data
        form = form_service.create_asset_form(
            ticker=ticker,
            quantity=position.quantity,
            avg_price=position.avg_price
        )
    
    return render_template('edit_asset.html', form=form, ticker=ticker, asset=position)

@web_bp.route('/portfolio/delete/<ticker>')
def delete_asset(ticker):
    """Delete an asset from portfolio."""
    portfolio = portfolio_service.load_portfolio()
    
    if ticker in portfolio.positions:
        del portfolio.positions[ticker]
        portfolio_service.save_portfolio(portfolio)
        
        # Clear cache
        cache_manager.delete(f"asset_analysis:{ticker}")
        cache_manager.delete(f"portfolio_analysis:{hash(str(portfolio))}")
        
        flash(f'Ativo {ticker} removido com sucesso!', 'success')
    else:
        flash(f'Ativo {ticker} não encontrado!', 'danger')
    
    return redirect(url_for('web.portfolio'))

@web_bp.route('/watchlist')
def watchlist():
    """Watchlist management page."""
    portfolio = portfolio_service.load_portfolio()
    form = form_service.create_watchlist_form()
    return render_template('watchlist.html', watchlist=portfolio.watchlist, form=form)

@web_bp.route('/watchlist/add', methods=['POST'])
def add_to_watchlist():
    """Add ticker to watchlist."""
    form = form_service.create_watchlist_form()
    
    if form.validate_on_submit():
        portfolio = portfolio_service.load_portfolio()
        ticker = form.ticker.data.upper()
        
        # Validate ticker
        ticker_valid = market_service.validate_ticker(ticker)
        
        if ticker_valid:
            portfolio.watchlist[ticker] = {
                'symbol': ticker,
                'monitor': form.monitor.data
            }
            portfolio_service.save_portfolio(portfolio)
            
            # Clear watchlist cache
            cache_manager.delete(f"watchlist_analysis:{hash(str(portfolio.watchlist))}")
            
            flash(f'Ticker {ticker} adicionado à watchlist!', 'success')
        else:
            flash(f'Ticker {ticker} inválido ou não encontrado!', 'danger')
    
    return redirect(url_for('web.watchlist'))

@web_bp.route('/watchlist/delete/<ticker>')
def delete_from_watchlist(ticker):
    """Delete ticker from watchlist."""
    portfolio = portfolio_service.load_portfolio()
    
    if ticker in portfolio.watchlist:
        del portfolio.watchlist[ticker]
        portfolio_service.save_portfolio(portfolio)
        
        # Clear watchlist cache
        cache_manager.delete(f"watchlist_analysis:{hash(str(portfolio.watchlist))}")
        
        flash(f'Ticker {ticker} removido da watchlist!', 'success')
    else:
        flash(f'Ticker {ticker} não encontrado na watchlist!', 'danger')
    
    return redirect(url_for('web.watchlist'))

@web_bp.route('/account', methods=['GET', 'POST'])
def account():
    """Account balance management."""
    portfolio = portfolio_service.load_portfolio()
    
    if request.method == 'POST':
        form = form_service.create_account_form()
        if form.validate_on_submit():
            old_balance = portfolio.account_balance
            portfolio.account_balance = form.balance.data
            portfolio_service.save_portfolio(portfolio)
            
            # Take snapshot if significant change
            if abs(form.balance.data - old_balance) > 1:
                portfolio_service.add_portfolio_snapshot(portfolio)
            
            flash('Saldo da conta atualizado com sucesso!', 'success')
            return redirect(url_for('web.account'))
    else:
        form = form_service.create_account_form(balance=portfolio.account_balance)
    
    return render_template('account.html', form=form, account_balance=portfolio.account_balance)

@web_bp.route('/analyze')
def analyze():
    """Portfolio analysis page."""
    try:
        portfolio = portfolio_service.load_portfolio()
        
        if not portfolio.positions:
            flash('Adicione ativos à sua carteira para análise.', 'warning')
            return redirect(url_for('web.index'))
        
        # Get analysis settings
        risk_profile = request.args.get('risk', 'medium')
        extended_hours = request.args.get('extended', 'false').lower() == 'true'
        
        # Execute analyses
        portfolio_analysis = portfolio_service.analyze_portfolio(
            portfolio, risk_profile, extended_hours
        )
        
        watchlist_analysis = portfolio_service.analyze_watchlist(
            portfolio, risk_profile, extended_hours
        )
        
        rebalance_plan = portfolio_service.generate_rebalance_plan(
            portfolio_analysis, watchlist_analysis, portfolio
        )
        
        return render_template(
            'analysis.html',
            portfolio_analysis=portfolio_analysis,
            watchlist_analysis=watchlist_analysis,
            rebalance_plan=rebalance_plan,
            risk_profile=risk_profile,
            extended_hours=extended_hours
        )
    
    except Exception as e:
        flash(f'Erro ao realizar análise: {str(e)}', 'danger')
        return redirect(url_for('web.index'))

@web_bp.route('/execute_rebalance', methods=['POST'])
def execute_rebalance():
    """Execute rebalance plan."""
    try:
        # Load portfolio
        portfolio = portfolio_service.load_portfolio()
        
        # Get risk profile
        risk_profile = request.form.get('risk_profile', 'medium')
        
        # Generate analysis and rebalance plan
        portfolio_analysis = portfolio_service.analyze_portfolio(
            portfolio, risk_profile, False
        )
        
        watchlist_analysis = portfolio_service.analyze_watchlist(
            portfolio, risk_profile, False
        )
        
        rebalance_plan = portfolio_service.generate_rebalance_plan(
            portfolio_analysis, watchlist_analysis, portfolio
        )
        
        # Execute rebalance plan
        updated_portfolio, result = portfolio_service.execute_rebalance_plan(
            portfolio, rebalance_plan
        )
        
        # Save updated portfolio
        portfolio_service.save_portfolio(updated_portfolio)
        
        # Take a snapshot
        portfolio_service.add_portfolio_snapshot(updated_portfolio)
        
        # Clear cache
        cache_manager.clear()
        
        # Flash results
        flash(f"Rebalanceamento executado: {len(result['sells_executed'])} vendas, {len(result['buys_executed'])} compras", 'success')
        
        if result['errors']:
            for error in result['errors']:
                flash(f"Erro: {error.get('ticker', '')} - {error.get('error', '')}", 'warning')
        
        return redirect(url_for('web.index'))
    except Exception as e:
        flash(f'Erro ao executar rebalanceamento: {str(e)}', 'danger')
        return redirect(url_for('web.analyze'))

@web_bp.route('/refresh_prices')
def refresh_prices():
    """Refresh current prices for all assets."""
    portfolio = portfolio_service.load_portfolio()
    
    if not portfolio.positions:
        flash('Nenhum ativo na carteira para atualizar.', 'warning')
        return redirect(url_for('web.index'))
    
    # Get tickers
    tickers = list(portfolio.positions.keys())
    
    # Get current prices
    try:
        prices = market_service.get_current_prices(tickers)
        
        # Update prices in portfolio
        for ticker, price in prices.items():
            if ticker in portfolio.positions:
                old_price = portfolio.positions[ticker].current_price
                portfolio.positions[ticker].current_price = price
                portfolio.positions[ticker].current_position = portfolio.positions[ticker].quantity * price
                
                # Calculate change
                change = (price / old_price - 1) * 100 if old_price > 0 else 0
                flash(f'{ticker}: ${old_price:.2f} → ${price:.2f} ({change:.2f}%)', 'info')
        
        portfolio_service.save_portfolio(portfolio)
        
        # Clear price cache
        for ticker in tickers:
            cache_manager.delete(f"current_price:{ticker}")
        
        flash('Preços atualizados com sucesso!', 'success')
    except Exception as e:
        flash(f'Erro ao atualizar preços: {str(e)}', 'danger')
    
    return redirect(url_for('web.index'))

@web_bp.route('/market_sentiment')
def market_sentiment():
    """Market sentiment page."""
    try:
        # Force refresh if requested
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        sentiment = market_service.get_market_sentiment(force_refresh=force_refresh)
        return render_template('market_sentiment.html', sentiment=sentiment)
    except Exception as e:
        flash(f'Erro ao obter sentimento de mercado: {str(e)}', 'danger')
        return redirect(url_for('web.index'))

@web_bp.route('/indicators', methods=['GET', 'POST'])
def indicators():
    """Technical indicators analysis page."""
    form = form_service.create_indicator_analysis_form()
    analysis_results = None
    charts = {}
    
    if request.method == 'POST' and form.validate_on_submit():
        ticker = form.ticker.data.upper()
        days = form.days.data
        interval = form.interval.data
        
        # Check cache first
        cache_key = f"indicator_analysis:{ticker}:{days}:{interval}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return render_template('indicators.html', form=form, 
                                results=cached_result.get('results'), 
                                charts=cached_result.get('charts', {}))
        
        # Validate ticker
        ticker_valid = market_service.validate_ticker(ticker)
        
        if not ticker_valid:
            flash(f'Ticker {ticker} inválido ou não encontrado!', 'danger')
            return render_template('indicators.html', form=form)
        
        try:
            # Get data
            df_dict = market_service.get_ticker_data(ticker, days, interval)
            
            if not df_dict or len(df_dict) < 5:
                flash(f'Dados insuficientes para {ticker}.', 'warning')
                return render_template('indicators.html', form=form)
            
            # Convert to DataFrame
            df = pd.DataFrame(df_dict)
            df.set_index('Date', inplace=True)
            
            # Add indicators
            df_indicators = TechnicalIndicators.add_all_indicators(df, {})
            
            # Calculate statistics
            stats = {
                'ticker': ticker,
                'current_price': float(df_indicators['Close'].iloc[-1]),
                'change_1d': float(df_indicators['Close'].iloc[-1] / df_indicators['Close'].iloc[-2] - 1) if len(df_indicators) > 1 else 0,
                'change_5d': float(df_indicators['Close'].iloc[-1] / df_indicators['Close'].iloc[-5] - 1) if len(df_indicators) > 4 else 0,
                'change_30d': float(df_indicators['Close'].iloc[-1] / df_indicators['Close'].iloc[-30] - 1) if len(df_indicators) > 29 else 0,
                'volume': float(df_indicators['Volume'].iloc[-1]) if 'Volume' in df_indicators.columns else 0,
                'rsi': float(df_indicators['RSI'].iloc[-1]) if 'RSI' in df_indicators.columns else 0,
                'macd': float(df_indicators['MACD_hist'].iloc[-1]) if 'MACD_hist' in df_indicators.columns else 0,
                'bb_position': (float(df_indicators['Close'].iloc[-1]) - float(df_indicators['BB_lower'].iloc[-1])) / (float(df_indicators['BB_upper'].iloc[-1]) - float(df_indicators['BB_lower'].iloc[-1])) if 'BB_lower' in df_indicators.columns and 'BB_upper' in df_indicators.columns else 0.5
            }
            
            # Generate charts using matplotlib
            # 1. Price with SMA and Bollinger Bands
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_indicators.index, df_indicators['Close'], label='Close')
            if 'SMA' in df_indicators.columns:
                ax.plot(df_indicators.index, df_indicators['SMA'], label='SMA')
            if 'BB_upper' in df_indicators.columns and 'BB_lower' in df_indicators.columns:
                ax.plot(df_indicators.index, df_indicators['BB_upper'], 'r--', label='Upper BB')
                ax.plot(df_indicators.index, df_indicators['BB_lower'], 'r--', label='Lower BB')
            ax.set_title(f'{ticker} Price Chart')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            
            # Save to buffer
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            charts['price'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            # 2. RSI Chart
            if 'RSI' in df_indicators.columns:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(df_indicators.index, df_indicators['RSI'])
                ax.axhline(y=70, color='r', linestyle='-')
                ax.axhline(y=30, color='g', linestyle='-')
                ax.axhline(y=50, color='k', linestyle='--')
                ax.set_title(f'{ticker} RSI Chart')
                ax.set_xlabel('Date')
                ax.set_ylabel('RSI')
                ax.grid(True)
                
                # Save to buffer
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png')
                buf.seek(0)
                charts['rsi'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)
            
            # 3. MACD Chart
            if 'MACD_line' in df_indicators.columns and 'MACD_signal' in df_indicators.columns:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(df_indicators.index, df_indicators['MACD_line'], label='MACD')
                ax.plot(df_indicators.index, df_indicators['MACD_signal'], label='Signal')
                ax.bar(df_indicators.index, df_indicators['MACD_hist'], color=[('g' if x > 0 else 'r') for x in df_indicators['MACD_hist']])
                ax.set_title(f'{ticker} MACD Chart')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True)
                
                # Save to buffer
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png')
                buf.seek(0)
                charts['macd'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)
            
            # Get technical score
            tech_score = Strategy.technical_score(df_indicators, "medium", {})
            
            # Cache results
            cache_results = {
                'results': {
                    'stats': stats,
                    'tech_score': tech_score
                },
                'charts': charts
            }
            cache_manager.set(cache_key, cache_results, 3600)  # 1 hour cache
            
            analysis_results = cache_results['results']
            
        except Exception as e:
            flash(f'Erro ao analisar indicadores para {ticker}: {str(e)}', 'danger')
    
    return render_template('indicators.html', form=form, results=analysis_results, charts=charts)

@web_bp.route('/fundamental/<ticker>')
def fundamental(ticker):
    """Fundamental analysis page for a ticker."""
    ticker = ticker.upper()
    
    # Check cache
    cache_key = f"fundamental_analysis:{ticker}"
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        return render_template('fundamental.html', result=cached_result, ticker=ticker)
    
    # Validate ticker
    ticker_valid = market_service.validate_ticker(ticker)
    if not ticker_valid:
        flash(f'Ticker {ticker} inválido ou não encontrado!', 'danger')
        return redirect(url_for('web.index'))
    
    try:
        # Get fundamental analysis
        result = FundamentalAnalysis.fundamental_score(ticker)
        
        # Cache result
        cache_manager.set(cache_key, result, 86400)  # 24 hour cache
        
        return render_template('fundamental.html', result=result, ticker=ticker)
    except Exception as e:
        flash(f'Erro ao analisar dados fundamentais para {ticker}: {str(e)}', 'danger')
        return redirect(url_for('web.index'))

@web_bp.route('/news/<ticker>')
def news(ticker):
    """News analysis page for a ticker."""
    ticker = ticker.upper()
    
    # Check cache
    cache_key = f"news_analysis:{ticker}"
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        return render_template('news.html', result=cached_result, ticker=ticker)
    
    # Validate ticker
    ticker_valid = market_service.validate_ticker(ticker)
    if not ticker_valid:
        flash(f'Ticker {ticker} inválido ou não encontrado!', 'danger')
        return redirect(url_for('web.index'))
    
    try:
        # Get news analysis
        result = QualitativeAnalysis.analyze_news_sentiment(ticker)
        
        # Cache result
        cache_manager.set(cache_key, result, 3600)  # 1 hour cache
        
        return render_template('news.html', result=result, ticker=ticker)
    except Exception as e:
        flash(f'Erro ao analisar notícias para {ticker}: {str(e)}', 'danger')
        return redirect(url_for('web.index'))

@web_bp.route('/goals', methods=['GET', 'POST'])
def goals():
    """Recovery goals management page."""
    portfolio = portfolio_service.load_portfolio()
    
    if request.method == 'POST':
        form = form_service.create_goals_form()
        if form.validate_on_submit():
            target = form.target_recovery.data
            days = form.days.data
            
            if target and days:
                portfolio.goals = RecoveryGoal(
                    target_recovery=target,
                    days=days,
                    start_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                portfolio_service.save_portfolio(portfolio)
                
                flash('Metas de recuperação definidas com sucesso!', 'success')
            else:
                # Clear goals
                portfolio.goals = None
                portfolio_service.save_portfolio(portfolio)
                
                flash('Metas de recuperação removidas!', 'info')
            
            return redirect(url_for('web.index'))
    else:
        # Pre-fill form if goals exist
        if portfolio.goals:
            form = form_service.create_goals_form(
                target=portfolio.goals.target_recovery,
                days=portfolio.goals.days
            )
        else:
            form = form_service.create_goals_form()
    
    # Calculate some stats for display
    history = portfolio_service.load_portfolio_history()
    stats = {}
    
    if history and len(history) > 1:
        first_entry = history[0]
        last_entry = history[-1]
        
        # Calculate absolute change
        absolute_change = last_entry.get('total_value', 0) - first_entry.get('total_value', 0)
        
        # Calculate percentage change
        if first_entry.get('total_value', 0) > 0:
            percentage_change = (absolute_change / first_entry.get('total_value', 0)) * 100
        else:
            percentage_change = 0
        
        # Days elapsed
        days_elapsed = (datetime.strptime(last_entry.get('date', datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d") - 
                        datetime.strptime(first_entry.get('date', datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d")).days
        
        stats = {
            'start_value': first_entry.get('total_value', 0),
            'current_value': last_entry.get('total_value', 0),
            'absolute_change': absolute_change,
            'percentage_change': percentage_change,
            'days_elapsed': days_elapsed
        }
    
    return render_template('goals.html', form=form, goals=portfolio.goals, stats=stats)

@web_bp.route('/parameters', methods=['GET', 'POST'])
def parameters():
    """Trading parameters management page."""
    trading_service = TradingService(
        {'DATA_FOLDER': os.environ.get('DATA_FOLDER', 'data')},
        cache_manager
    )
    
    if request.method == 'POST':
        form = form_service.create_parameters_form()
        if form.validate_on_submit():
            params = {
                'sma_period': form.sma_period.data,
                'ema_period': form.ema_period.data,
                'rsi_period': form.rsi_period.data,
                'macd_fast': form.macd_fast.data,
                'macd_slow': form.macd_slow.data,
                'macd_signal': form.macd_signal.data,
                'bb_window': form.bb_window.data,
                'bb_std': form.bb_std.data,
                'decision_buy_threshold': form.decision_buy_threshold.data,
                'decision_sell_threshold': form.decision_sell_threshold.data,
                'take_profit_pct': form.take_profit_pct.data,
                'stop_loss_pct': form.stop_loss_pct.data,
                'trailing_stop_pct': form.trailing_stop_pct.data
            }
            
            # Update parameters
            result = trading_service.update_trading_parameters(params)
            
            if result.get('success'):
                flash('Parâmetros de trading atualizados com sucesso!', 'success')
                
                # Clear analysis cache
                cache_manager.delete("portfolio_analysis:")
                cache_manager.delete("watchlist_analysis:")
            else:
                flash(f'Erro ao atualizar parâmetros: {result.get("message")}', 'danger')
            
            return redirect(url_for('web.parameters'))
    else:
        # Get current parameters
        params = trading_service.get_trading_parameters()
        
        # Create form with current parameters
        form = form_service.create_parameters_form(params)
    
    return render_template('parameters.html', form=form, params=params)

@web_bp.route('/history')
def history():
    """Portfolio history page."""
    # Get history data
    history_data = portfolio_service.load_portfolio_history()
    
    if not history_data:
        flash('Sem histórico disponível ainda.', 'info')
        return redirect(url_for('web.index'))
    
    # Process history data for plotting
    dates = []
    values = []
    cash_values = []
    portfolio_values = []
    
    for entry in history_data:
        dates.append(entry.get('date'))
        values.append(entry.get('total_value', 0))
        cash_values.append(entry.get('cash_balance', 0))
        portfolio_values.append(entry.get('portfolio_value', 0))
    
    # Generate chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, values, label='Total Value')
    ax.plot(dates, portfolio_values, label='Portfolio Value')
    ax.plot(dates, cash_values, label='Cash')
    ax.set_title('Portfolio History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value ($)')
    ax.legend()
    ax.grid(True)
    
    # Save to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    history_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return render_template('history.html', history=history_data, chart=history_chart)

@web_bp.route('/cache')
def cache():
    """Cache management page."""
    # Get cache status
    cache_status = cache_manager.get_cache_status() if hasattr(cache_manager, 'get_cache_status') else {
        'cache_size': len(cache_manager.keys()),
        'memory_usage_mb': 'N/A',
        'items_count': len(cache_manager.keys())
    }
    
    return render_template('cache.html', status=cache_status)

@web_bp.route('/cache/clear')
def clear_cache():
    """Clear all cache."""
    try:
        cache_manager.clear()
        flash('Cache limpo com sucesso!', 'success')
    except Exception as e:
        flash(f'Erro ao limpar cache: {str(e)}', 'danger')
    
    return redirect(url_for('web.cache'))

@web_bp.route('/diagnostics')
def diagnostics():
    """System diagnostics page."""
    # Create system service
    system_service = SystemService({'DATA_FOLDER': os.environ.get('DATA_FOLDER', 'data')})
    
    # Run diagnostics
    diagnostics_result = system_service.run_diagnostics()
    
    # Check for updates
    update_info = system_service.check_for_updates()
    
    return render_template('diagnostics.html', 
                          diagnostics=diagnostics_result, 
                          update_info=update_info)

@web_bp.route('/help')
def help():
    """Help and documentation page."""
    return render_template('help.html')

# Initialize the blueprint
web_blueprint = web_bp