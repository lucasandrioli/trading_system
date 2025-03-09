import os
import json
import importlib
import traceback
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_cors import CORS
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, FloatField, IntegerField, BooleanField, SubmitField, SelectField, TextAreaField
from wtforms.validators import DataRequired, Optional, NumberRange
from io import BytesIO
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Importar o sistema de trading existente
import trading_system
from trading_system import (
    DataLoader, TechnicalIndicators, MarketAnalysis, Strategy, APIClient,
    FundamentalAnalysis, QualitativeAnalysis, analyze_portfolio,
    analyze_watchlist, generate_rebalance_plan, DYNAMIC_PARAMS,
    PortfolioExecutor, PortfolioOptimizer, RecoveryTracker,
    NotificationSystem, create_backtest, calculate_performance_metrics,
    check_for_updates, run_diagnostics, VERSION
)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'trading-system-secret-key')
app.config['DATA_FOLDER'] = os.environ.get('DATA_FOLDER', 'data')
app.config['POLYGON_API_KEY'] = os.environ.get('POLYGON_API_KEY', '')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
csrf = CSRFProtect(app)
CORS(app)

# Garantir que o diretório de dados existe
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
PORTFOLIO_FILE = os.path.join(app.config['DATA_FOLDER'], 'portfolio.json')
HISTORY_FILE = os.path.join(app.config['DATA_FOLDER'], 'portfolio_history.json')
NOTIFICATION_CONFIG = os.path.join(app.config['DATA_FOLDER'], 'notification_config.json')
TRAILING_DATA_FILE = os.path.join(app.config['DATA_FOLDER'], 'trailing_data.json')

# Initialize notification system
def get_notification_config():
    if os.path.exists(NOTIFICATION_CONFIG):
        try:
            with open(NOTIFICATION_CONFIG, 'r') as f:
                return json.load(f)
        except Exception as e:
            app.logger.error(f"Error loading notification config: {e}")
    return {"enabled": False, "methods": {"email": {"enabled": False}, "webhook": {"enabled": False}}}

notification_system = NotificationSystem(get_notification_config())

# Load trailing stop data
def load_trailing_data():
    if os.path.exists(TRAILING_DATA_FILE):
        try:
            with open(TRAILING_DATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            app.logger.error(f"Error loading trailing data: {e}")
    return {}

def save_trailing_data(data):
    try:
        with open(TRAILING_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        app.logger.error(f"Error saving trailing data: {e}")
        return False

# Carregar dados do portfólio
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            app.logger.error(f"Erro ao carregar portfólio: {e}")
            return {"portfolio": {}, "watchlist": {}, "account_balance": 0.0, "goals": {}}
    else:
        return {"portfolio": {}, "watchlist": {}, "account_balance": 0.0, "goals": {}}

# Salvar dados do portfólio
def save_portfolio(data):
    try:
        os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        app.logger.error(f"Erro ao salvar portfólio: {e}")
        return False

# Load portfolio history
def load_portfolio_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            app.logger.error(f"Error loading portfolio history: {e}")
            return []
    return []

def save_portfolio_history(data):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        app.logger.error(f"Error saving portfolio history: {e}")
        return False

def add_portfolio_snapshot():
    data = load_portfolio()
    portfolio = data.get('portfolio', {})
    account_balance = data.get('account_balance', 0)
    
    # Calculate total value
    total_current = sum(details.get('quantity', 0) * details.get('current_price', 0) 
                       for ticker, details in portfolio.items())
    
    # Create snapshot
    snapshot = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": total_current,
        "cash_balance": account_balance,
        "total_value": total_current + account_balance,
        "positions": {ticker: {
            "quantity": details.get('quantity', 0),
            "price": details.get('current_price', 0),
            "value": details.get('quantity', 0) * details.get('current_price', 0)
        } for ticker, details in portfolio.items()}
    }
    
    # Add to history
    history = load_portfolio_history()
    history.append(snapshot)
    
    # Keep last 90 days
    if len(history) > 90:
        history = history[-90:]
    
    save_portfolio_history(history)
    return snapshot

# Formulários
class AssetForm(FlaskForm):
    ticker = StringField('Ticker', validators=[DataRequired()])
    quantity = IntegerField('Quantidade', validators=[DataRequired(), NumberRange(min=1)])
    avg_price = FloatField('Preço Médio', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Salvar')

class WatchlistForm(FlaskForm):
    ticker = StringField('Ticker', validators=[DataRequired()])
    monitor = BooleanField('Monitorar', default=True)
    submit = SubmitField('Adicionar')

class AccountForm(FlaskForm):
    balance = FloatField('Saldo da Conta', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Atualizar')

class GoalsForm(FlaskForm):
    target_recovery = FloatField('Meta de Recuperação ($)', validators=[Optional(), NumberRange(min=0)])
    days = IntegerField('Prazo (dias)', validators=[Optional(), NumberRange(min=1)])
    submit = SubmitField('Definir Metas')

class TradeForm(FlaskForm):
    ticker = StringField('Ticker', validators=[DataRequired()])
    quantity = IntegerField('Quantidade', validators=[DataRequired(), NumberRange(min=1)])
    price = FloatField('Preço', validators=[DataRequired(), NumberRange(min=0.01)])
    trade_type = StringField('Tipo (COMPRA/VENDA)', validators=[DataRequired()])
    submit = SubmitField('Registrar')

class BacktestForm(FlaskForm):
    days = IntegerField('Período (dias)', validators=[DataRequired(), NumberRange(min=5, max=365)])
    benchmark = StringField('Benchmark', validators=[DataRequired()], default='SPY')
    risk_profile = SelectField('Perfil de Risco', choices=[
        ('low', 'Baixo'), 
        ('medium', 'Médio'), 
        ('high', 'Alto'), 
        ('ultra', 'Ultra')
    ], default='medium')
    submit = SubmitField('Executar Backtest')

class OptimizationForm(FlaskForm):
    max_positions = IntegerField('Máximo de Posições', validators=[DataRequired(), NumberRange(min=1, max=50)], default=15)
    cash_reserve_pct = FloatField('Reserva de Caixa (%)', validators=[DataRequired(), NumberRange(min=0, max=50)], default=10)
    max_position_size_pct = FloatField('Tamanho Máximo de Posição (%)', validators=[DataRequired(), NumberRange(min=1, max=50)], default=20)
    min_score = FloatField('Score Mínimo para Inclusão', validators=[DataRequired(), NumberRange(min=0, max=100)], default=30)
    risk_profile = SelectField('Perfil de Risco', choices=[
        ('low', 'Baixo'), 
        ('medium', 'Médio'), 
        ('high', 'Alto'), 
        ('ultra', 'Ultra')
    ], default='medium')
    submit = SubmitField('Otimizar Carteira')

class NotificationForm(FlaskForm):
    enabled = BooleanField('Ativar Notificações', default=False)
    email_enabled = BooleanField('Notificações por Email', default=False)
    email_address = StringField('Endereço de Email')
    webhook_enabled = BooleanField('Webhook (Discord/Slack)', default=False)
    webhook_url = StringField('URL do Webhook')
    notify_trades = BooleanField('Notificar Operações', default=True)
    notify_thresholds = BooleanField('Notificar Limiares de Preço', default=True)
    submit = SubmitField('Salvar Configurações')

class IndicatorAnalysisForm(FlaskForm):
    ticker = StringField('Ticker', validators=[DataRequired()])
    days = IntegerField('Período (dias)', validators=[DataRequired(), NumberRange(min=5, max=365)], default=60)
    interval = SelectField('Intervalo', choices=[
        ('1d', 'Diário'),
        ('1h', 'Horário'),
        ('1wk', 'Semanal')
    ], default='1d')
    submit = SubmitField('Analisar')

class FundamentalAnalysisForm(FlaskForm):
    ticker = StringField('Ticker', validators=[DataRequired()])
    submit = SubmitField('Analisar')

class NewsAnalysisForm(FlaskForm):
    ticker = StringField('Ticker', validators=[DataRequired()])
    submit = SubmitField('Buscar Notícias')

class TradingParamsForm(FlaskForm):
    sma_period = IntegerField('SMA Period', validators=[NumberRange(min=1, max=200)], default=20)
    ema_period = IntegerField('EMA Period', validators=[NumberRange(min=1, max=200)], default=9)
    rsi_period = IntegerField('RSI Period', validators=[NumberRange(min=1, max=50)], default=14)
    macd_fast = IntegerField('MACD Fast', validators=[NumberRange(min=1, max=50)], default=12)
    macd_slow = IntegerField('MACD Slow', validators=[NumberRange(min=1, max=50)], default=26)
    macd_signal = IntegerField('MACD Signal', validators=[NumberRange(min=1, max=50)], default=9)
    bb_window = IntegerField('Bollinger Window', validators=[NumberRange(min=1, max=100)], default=20)
    bb_std = FloatField('Bollinger Std', validators=[NumberRange(min=0.5, max=4)], default=2)
    decision_buy_threshold = FloatField('Buy Threshold', validators=[NumberRange(min=0, max=100)], default=60)
    decision_sell_threshold = FloatField('Sell Threshold', validators=[NumberRange(min=-100, max=0)], default=-60)
    take_profit_pct = FloatField('Take Profit %', validators=[NumberRange(min=0.5, max=50)], default=5.0)
    stop_loss_pct = FloatField('Stop Loss %', validators=[NumberRange(min=-50, max=-0.5)], default=-8.0)
    trailing_stop_pct = FloatField('Trailing Stop %', validators=[NumberRange(min=0.5, max=20)], default=3.0)
    submit = SubmitField('Atualizar Parâmetros')

# Rotas
@app.route('/')
def index():
    data = load_portfolio()
    
    # Calcular valores totais
    portfolio = data.get('portfolio', {})
    account_balance = data.get('account_balance', 0)
    
    total_invested = 0
    total_current = 0
    
    for ticker, details in portfolio.items():
        quantity = details.get('quantity', 0)
        avg_price = details.get('avg_price', 0)
        current_price = details.get('current_price', avg_price)
        
        invested = quantity * avg_price
        current = quantity * current_price
        
        total_invested += invested
        total_current += current
    
    portfolio_value = total_current
    total_value = portfolio_value + account_balance
    profit_loss = total_current - total_invested
    profit_loss_pct = (profit_loss / total_invested * 100) if total_invested > 0 else 0
    
    # Get market sentiment
    try:
        market_sentiment = MarketAnalysis.get_market_sentiment()
    except Exception as e:
        app.logger.error(f"Error getting market sentiment: {e}")
        market_sentiment = {"sentiment": "unknown", "trend": "unknown", "market_bias_score": 0}
    
    # Get recovery tracker status
    goals = data.get('goals', {})
    recovery_status = {}
    if goals and goals.get('target_recovery', 0) > 0:
        history = load_portfolio_history()
        if history:
            first_value = history[0]['total_value'] if history else total_value
            current_pnl = total_value - first_value
            progress_pct = (current_pnl / goals.get('target_recovery', 1)) * 100
            
            # Calculate days passed
            start_date = datetime.strptime(goals.get('start_date', datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d')
            days_passed = (datetime.now() - start_date).days
            days_total = goals.get('days', 30)
            days_remaining = max(0, days_total - days_passed)
            
            recovery_status = {
                'current_pnl': current_pnl,
                'progress_pct': progress_pct,
                'days_passed': days_passed,
                'days_remaining': days_remaining,
                'daily_target': goals.get('target_recovery', 0) / max(1, days_remaining) if days_remaining > 0 else 0,
                'on_track': progress_pct >= (days_passed / max(1, days_total) * 100) if days_total > 0 else False
            }
    
    summary = {
        'total_invested': total_invested,
        'portfolio_value': portfolio_value,
        'account_balance': account_balance,
        'total_value': total_value,
        'profit_loss': profit_loss,
        'profit_loss_pct': profit_loss_pct,
        'position_count': len(portfolio),
        'market_sentiment': market_sentiment,
        'recovery_status': recovery_status
    }
    
    # Take a daily snapshot
    last_history = load_portfolio_history()
    if not last_history or last_history[-1]['date'] != datetime.now().strftime("%Y-%m-%d"):
        add_portfolio_snapshot()
    
    # Check for updates
    update_info = check_for_updates(VERSION)
    
    return render_template(
        'index.html', 
        portfolio=portfolio, 
        watchlist=data.get('watchlist', {}),
        summary=summary,
        goals=goals,
        update_info=update_info,
        version=VERSION
    )

@app.route('/portfolio')
def portfolio():
    data = load_portfolio()
    form = AssetForm()
    return render_template('portfolio.html', portfolio=data.get('portfolio', {}), form=form)

@app.route('/portfolio/add', methods=['POST'])
def add_asset():
    form = AssetForm()
    if form.validate_on_submit():
        data = load_portfolio()
        ticker = form.ticker.data.upper()
        
        # Validar ticker
        if DataLoader.check_ticker_valid(ticker):
            # Obter preço atual
            df = DataLoader.get_asset_data(ticker, days=1)
            current_price = float(df['Close'].iloc[-1]) if not df.empty and 'Close' in df.columns else form.avg_price.data
            
            # Check if this is an update to an existing position
            is_update = ticker in data['portfolio']
            
            if is_update:
                # For updates, only consider the additional investment if any
                old_qty = data['portfolio'][ticker].get('quantity', 0)
                old_price = data['portfolio'][ticker].get('avg_price', 0)
                new_qty = form.quantity.data
                new_price = form.avg_price.data
                
                # Calculate only the additional investment
                additional_investment = 0
                if new_qty > old_qty:
                    # Only check balance for additional shares
                    additional_qty = new_qty - old_qty
                    additional_investment = additional_qty * new_price
                    
                    # Check if sufficient balance for additional investment
                    if additional_investment > data.get('account_balance', 0):
                        flash(f'Saldo insuficiente para adicionar {additional_qty} ações. Necessário: ${additional_investment:.2f}, Disponível: ${data.get("account_balance", 0):.2f}', 'warning')
                        # Allow the update to proceed anyway - user is editing existing position
                
                # Update the position with new values
                total_qty = new_qty
                avg_price = ((old_qty * old_price) + ((new_qty - old_qty) * new_price)) / total_qty if total_qty > 0 else 0
                
                # If reducing position size, return money to account
                if new_qty < old_qty:
                    reduced_qty = old_qty - new_qty
                    returned_funds = reduced_qty * old_price
                    data['account_balance'] += returned_funds
                # If increasing position size, deduct from account
                elif new_qty > old_qty:
                    data['account_balance'] -= additional_investment
                
                data['portfolio'][ticker] = {
                    'symbol': ticker,
                    'quantity': total_qty,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'current_position': total_qty * current_price,
                    'last_buy': {
                        'price': form.avg_price.data,
                        'quantity': form.quantity.data,
                        'date': datetime.now().isoformat()
                    }
                }
                flash(f'Ativo {ticker} atualizado com sucesso!', 'success')
            else:
                # For new positions, check available balance
                investment_amount = form.quantity.data * form.avg_price.data
                
                # Check if sufficient balance
                if investment_amount > data.get('account_balance', 0):
                    flash(f'Saldo insuficiente para comprar {form.quantity.data} ações. Necessário: ${investment_amount:.2f}, Disponível: ${data.get("account_balance", 0):.2f}', 'danger')
                    return redirect(url_for('portfolio'))
                
                # Deduct from account balance
                data['account_balance'] -= investment_amount
                
                # Add new position
                data['portfolio'][ticker] = {
                    'symbol': ticker,
                    'quantity': form.quantity.data,
                    'avg_price': form.avg_price.data,
                    'current_price': current_price,
                    'current_position': form.quantity.data * current_price,
                    'last_buy': {
                        'price': form.avg_price.data,
                        'quantity': form.quantity.data,
                        'date': datetime.now().isoformat()
                    },
                    'last_sell': None
                }
                flash(f'Ativo {ticker} adicionado com sucesso!', 'success')
            
            save_portfolio(data)
            
            # Take a snapshot after portfolio change
            add_portfolio_snapshot()
            
            # Send notification
            notification_system.alert_trade_recommendation(
                ticker, "COMPRA" if not is_update else "ATUALIZAÇÃO", form.quantity.data, 
                form.avg_price.data, "Adicionado/atualizado manualmente à carteira"
            )
        else:
            flash(f'Ticker {ticker} inválido ou não encontrado!', 'danger')
        
    return redirect(url_for('portfolio'))

@app.route('/portfolio/edit/<ticker>', methods=['GET', 'POST'])
def edit_asset(ticker):
    data = load_portfolio()
    
    if ticker not in data['portfolio']:
        flash(f'Ativo {ticker} não encontrado na carteira!', 'danger')
        return redirect(url_for('portfolio'))
    
    form = AssetForm()
    
    if request.method == 'POST' and form.validate_on_submit():
        new_qty = form.quantity.data
        new_price = form.avg_price.data
        
        # Obter preço atual
        df = DataLoader.get_asset_data(ticker, days=1)
        current_price = float(df['Close'].iloc[-1]) if not df.empty and 'Close' in df.columns else new_price
        
        # Atualizar posição
        data['portfolio'][ticker]['quantity'] = new_qty
        data['portfolio'][ticker]['avg_price'] = new_price
        data['portfolio'][ticker]['current_price'] = current_price
        data['portfolio'][ticker]['current_position'] = new_qty * current_price
        
        save_portfolio(data)
        add_portfolio_snapshot()
        
        flash(f'Ativo {ticker} atualizado com sucesso!', 'success')
        return redirect(url_for('portfolio'))
    
    # Preencher formulário com dados existentes
    if request.method == 'GET':
        position = data['portfolio'][ticker]
        form.ticker.data = ticker
        form.quantity.data = position.get('quantity', 0)
        form.avg_price.data = position.get('avg_price', 0)
    
    return render_template('edit_asset.html', form=form, ticker=ticker, asset=data['portfolio'][ticker])

@app.route('/portfolio/delete/<ticker>')
def delete_asset(ticker):
    data = load_portfolio()
    if ticker in data['portfolio']:
        del data['portfolio'][ticker]
        save_portfolio(data)
        flash(f'Ativo {ticker} removido com sucesso!', 'success')
    else:
        flash(f'Ativo {ticker} não encontrado!', 'danger')
    
    return redirect(url_for('portfolio'))

@app.route('/watchlist')
def watchlist():
    data = load_portfolio()
    form = WatchlistForm()
    return render_template('watchlist.html', watchlist=data.get('watchlist', {}), form=form)

@app.route('/watchlist/add', methods=['POST'])
def add_to_watchlist():
    form = WatchlistForm()
    if form.validate_on_submit():
        data = load_portfolio()
        ticker = form.ticker.data.upper()
        
        # Validar ticker
        if DataLoader.check_ticker_valid(ticker):
            data['watchlist'][ticker] = {
                'symbol': ticker,
                'monitor': form.monitor.data
            }
            save_portfolio(data)
            flash(f'Ticker {ticker} adicionado à watchlist!', 'success')
        else:
            flash(f'Ticker {ticker} inválido ou não encontrado!', 'danger')
    
    return redirect(url_for('watchlist'))

@app.route('/watchlist/delete/<ticker>')
def delete_from_watchlist(ticker):
    data = load_portfolio()
    if ticker in data['watchlist']:
        del data['watchlist'][ticker]
        save_portfolio(data)
        flash(f'Ticker {ticker} removido da watchlist!', 'success')
    else:
        flash(f'Ticker {ticker} não encontrado na watchlist!', 'danger')
    
    return redirect(url_for('watchlist'))

@app.route('/account', methods=['GET', 'POST'])
def account():
    data = load_portfolio()
    form = AccountForm()
    
    if request.method == 'POST' and form.validate_on_submit():
        old_balance = data.get('account_balance', 0)
        data['account_balance'] = form.balance.data
        save_portfolio(data)
        
        # If significant change, log it in history
        if abs(form.balance.data - old_balance) > 1:
            add_portfolio_snapshot()
            
        flash('Saldo da conta atualizado com sucesso!', 'success')
        return redirect(url_for('account'))
    
    form.balance.data = data.get('account_balance', 0)
    return render_template('account.html', form=form, account_balance=data.get('account_balance', 0))

@app.route('/goals', methods=['GET', 'POST'])
def goals():
    data = load_portfolio()
    form = GoalsForm()
    
    if request.method == 'POST' and form.validate_on_submit():
        data['goals'] = {
            'target_recovery': form.target_recovery.data,
            'days': form.days.data,
            'start_date': datetime.now().strftime('%Y-%m-%d')
        }
        save_portfolio(data)
        # Take a snapshot when goals are set
        add_portfolio_snapshot()
        flash('Metas atualizadas com sucesso!', 'success')
        return redirect(url_for('goals'))
    
    if data.get('goals'):
        form.target_recovery.data = data['goals'].get('target_recovery', 0)
        form.days.data = data['goals'].get('days', 0)
    
    # Get recovery tracker data
    history = load_portfolio_history()
    recovery_data = None
    if data.get('goals') and data['goals'].get('target_recovery', 0) > 0 and history:
        # Calculate portfolio values
        portfolio = data.get('portfolio', {})
        account_balance = data.get('account_balance', 0)
        total_current = sum(details.get('quantity', 0) * details.get('current_price', 0) 
                           for ticker, details in portfolio.items())
        total_value = total_current + account_balance
        
        # Get initial value and days info
        start_date = datetime.strptime(data['goals'].get('start_date', datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d')
        days_passed = (datetime.now() - start_date).days
        days_total = data['goals'].get('days', 30)
        days_remaining = max(0, days_total - days_passed)
        
        # Get initial value
        initial_value = history[0]['total_value'] if history else total_value
        current_pnl = total_value - initial_value
        progress_pct = (current_pnl / data['goals'].get('target_recovery', 1)) * 100
        
        recovery_data = {
            'initial_value': initial_value,
            'current_value': total_value,
            'current_pnl': current_pnl,
            'target': data['goals'].get('target_recovery', 0),
            'progress_pct': progress_pct,
            'days_passed': days_passed,
            'days_total': days_total,
            'days_remaining': days_remaining,
            'daily_target': data['goals'].get('target_recovery', 0) / max(1, days_remaining) if days_remaining > 0 else 0,
            'on_track': progress_pct >= (days_passed / max(1, days_total) * 100) if days_total > 0 else False,
            'history': history
        }
    
    return render_template('goals.html', form=form, goals=data.get('goals', {}), recovery_data=recovery_data)

@app.route('/trade', methods=['GET', 'POST'])
def trade():
    data = load_portfolio()
    form = TradeForm()
    
    if request.method == 'POST' and form.validate_on_submit():
        ticker = form.ticker.data.upper()
        quantity = form.quantity.data
        price = form.price.data
        trade_type = form.trade_type.data.upper()
        
        if trade_type not in ["COMPRA", "VENDA"]:
            flash('Tipo de operação inválido. Use COMPRA ou VENDA.', 'danger')
            return redirect(url_for('trade'))
        
        # Cálculo da comissão
        operation_value = quantity * price
        commission = trading_system.calculate_xp_commission(operation_value)
        
        if trade_type == "COMPRA":
            # Verificar se é uma posição existente
            is_existing = ticker in data['portfolio']
            
            if is_existing:
                # Para posições existentes
                old_qty = data['portfolio'][ticker].get('quantity', 0)
                old_price = data['portfolio'][ticker].get('avg_price', 0)
                
                # Atualizar portfólio
                new_qty = old_qty + quantity
                avg_price = ((old_qty * old_price) + (quantity * price)) / new_qty if new_qty > 0 else 0
                
                # Verificar saldo apenas para novas ações
                if operation_value + commission > data.get('account_balance', 0):
                    flash('Saldo insuficiente para esta operação!', 'danger')
                    return redirect(url_for('trade'))
                
                data['portfolio'][ticker]['quantity'] = new_qty
                data['portfolio'][ticker]['avg_price'] = avg_price
                data['portfolio'][ticker]['current_price'] = price
                data['portfolio'][ticker]['current_position'] = new_qty * price
                data['portfolio'][ticker]['last_buy'] = {
                    'price': price,
                    'quantity': quantity,
                    'date': datetime.now().isoformat()
                }
            else:
                # Para novas posições
                if operation_value + commission > data.get('account_balance', 0):
                    flash('Saldo insuficiente para esta operação!', 'danger')
                    return redirect(url_for('trade'))
                
                data['portfolio'][ticker] = {
                    'symbol': ticker,
                    'quantity': quantity,
                    'avg_price': price,
                    'current_price': price,
                    'current_position': quantity * price,
                    'last_buy': {
                        'price': price,
                        'quantity': quantity,
                        'date': datetime.now().isoformat()
                    },
                    'last_sell': None
                }
            
            # Atualizar saldo
            data['account_balance'] -= (operation_value + commission)
            flash(f'Compra de {quantity} {ticker} a ${price:.2f} registrada com sucesso!', 'success')
            
            # Send notification
            notification_system.alert_execution({
                "action": "COMPRA",
                "ticker": ticker,
                "quantity": quantity,
                "price": price,
                "gross_value": operation_value,
                "commission": commission
            })
        
        elif trade_type == "VENDA":
            # Verificar se possui o ativo e a quantidade
            if ticker not in data['portfolio']:
                flash(f'Você não possui {ticker} em sua carteira!', 'danger')
                return redirect(url_for('trade'))
            
            current_qty = data['portfolio'][ticker].get('quantity', 0)
            
            # Se a quantidade for exatamente igual à existente, considera-se venda completa
            # Se for menor, considera-se venda parcial
            if quantity > current_qty:
                flash(f'Quantidade insuficiente. Você possui {current_qty} {ticker}.', 'danger')
                return redirect(url_for('trade'))
            
            # Atualizar portfólio
            new_qty = current_qty - quantity
            data['portfolio'][ticker]['quantity'] = new_qty
            data['portfolio'][ticker]['current_price'] = price
            data['portfolio'][ticker]['current_position'] = new_qty * price
            data['portfolio'][ticker]['last_sell'] = {
                'price': price,
                'quantity': quantity,
                'date': datetime.now().isoformat()
            }
            
            # Se quantidade = 0, remover ativo
            if new_qty <= 0:
                del data['portfolio'][ticker]
            
            # Atualizar saldo
            data['account_balance'] += (operation_value - commission)
            flash(f'Venda de {quantity} {ticker} a ${price:.2f} registrada com sucesso!', 'success')
            
            # Send notification
            notification_system.alert_execution({
                "action": "VENDA",
                "ticker": ticker,
                "quantity": quantity,
                "price": price,
                "gross_value": operation_value,
                "commission": commission
            })
        
        save_portfolio(data)
        # Take a snapshot after trade
        add_portfolio_snapshot()
        return redirect(url_for('index'))
    
    # Pre-fill form from query parameters
    if request.method == 'GET':
        form.ticker.data = request.args.get('ticker', '')
        form.quantity.data = request.args.get('quantity', None)
        form.price.data = request.args.get('price', None)
        form.trade_type.data = request.args.get('trade_type', '')
    
    return render_template('trade.html', form=form)

@app.route('/refresh_prices')
def refresh_prices():
    data = load_portfolio()
    portfolio = data.get('portfolio', {})
    
    if not portfolio:
        flash('Nenhum ativo na carteira para atualizar.', 'warning')
        return redirect(url_for('index'))
    
    # Obter todos os tickers
    tickers = list(portfolio.keys())
    
    # Obter preços atuais
    try:
        prices = DataLoader.get_realtime_prices_bulk(tickers)
        
        # Atualizar preços no portfólio
        for ticker, price in prices.items():
            if ticker in portfolio:
                old_price = portfolio[ticker].get('current_price', 0)
                portfolio[ticker]['current_price'] = price
                portfolio[ticker]['current_position'] = portfolio[ticker]['quantity'] * price
                
                # Calcular variação
                change = (price / old_price - 1) * 100 if old_price > 0 else 0
                flash(f'{ticker}: ${old_price:.2f} → ${price:.2f} ({change:.2f}%)', 'info')
        
        save_portfolio(data)
        flash('Preços atualizados com sucesso!', 'success')
    except Exception as e:
        app.logger.error(f"Erro ao atualizar preços: {e}")
        flash(f'Erro ao atualizar preços: {str(e)}', 'danger')
    
    return redirect(url_for('index'))

@app.route('/analyze')
def analyze():
    try:
        data = load_portfolio()
        portfolio = data.get('portfolio', {})
        watchlist = data.get('watchlist', {})
        account_balance = data.get('account_balance', 0)
        goals = data.get('goals', {})
        
        if not portfolio:
            flash('Adicione ativos à sua carteira para análise.', 'warning')
            return redirect(url_for('index'))
        
        # Configurações para análise
        risk_profile = request.args.get('risk', 'medium')
        trailing_data = load_trailing_data()
        extended_hours = request.args.get('extended', 'false').lower() == 'true'
        
        # Executar análises
        portfolio_analysis = analyze_portfolio(
            portfolio, account_balance, risk_profile,
            trailing_data, extended_hours, goals
        )
        
        watchlist_analysis = analyze_watchlist(
            watchlist, account_balance, risk_profile,
            extended_hours, goals
        )
        
        rebalance_plan = generate_rebalance_plan(
            portfolio_analysis, watchlist_analysis,
            account_balance, DYNAMIC_PARAMS
        )
        
        # Save trailing data if updated
        save_trailing_data(trailing_data)
        
        # Guardar análise para uso na página
        return render_template(
            'analysis.html',
            portfolio_analysis=portfolio_analysis,
            watchlist_analysis=watchlist_analysis,
            rebalance_plan=rebalance_plan,
            risk_profile=risk_profile,
            extended_hours=extended_hours
        )
    
    except Exception as e:
        app.logger.error(f"Erro na análise: {e}")
        app.logger.error(traceback.format_exc())
        flash(f'Erro ao realizar análise: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/apply_rebalance')
def apply_rebalance():
    try:
        transaction_id = request.args.get('id')
        action = request.args.get('action')
        
        if not transaction_id or not action:
            flash('Parâmetros inválidos para rebalanceamento.', 'danger')
            return redirect(url_for('analyze'))
        
        # Here you would implement the logic to automatically apply
        # the rebalancing transaction based on ID and action
        data = load_portfolio()
        portfolio = data.get('portfolio', {})
        account_balance = data.get('account_balance', 0)
        
        # Create PortfolioExecutor instance
        executor = PortfolioExecutor(portfolio, account_balance)
        
        # For now, just redirect to manual trade screen with params
        return redirect(url_for('trade', ticker=transaction_id, trade_type=action))
    
    except Exception as e:
        app.logger.error(f"Erro ao aplicar rebalanceamento: {e}")
        flash(f'Erro ao aplicar rebalanceamento: {str(e)}', 'danger')
        return redirect(url_for('analyze'))

@app.route('/execute_rebalance', methods=['POST'])
def execute_rebalance():
    try:
        data = load_portfolio()
        portfolio = data.get('portfolio', {})
        account_balance = data.get('account_balance', 0)
        
        # Get risk profile
        risk_profile = request.form.get('risk_profile', 'medium')
        
        # Generate analysis and rebalance plan
        portfolio_analysis = analyze_portfolio(
            portfolio, account_balance, risk_profile,
            load_trailing_data(), False, data.get('goals', {})
        )
        
        watchlist_analysis = analyze_watchlist(
            data.get('watchlist', {}), account_balance, risk_profile,
            False, data.get('goals', {})
        )
        
        rebalance_plan = generate_rebalance_plan(
            portfolio_analysis, watchlist_analysis,
            account_balance, DYNAMIC_PARAMS
        )
        
        # Create PortfolioExecutor instance
        executor = PortfolioExecutor(portfolio, account_balance)
        
        # Execute rebalance plan
        result = executor.execute_rebalance_plan(rebalance_plan)
        
        # Update portfolio data
        data['portfolio'] = result['updated_portfolio']['positions']
        data['account_balance'] = result['ending_balance']
        save_portfolio(data)
        
        # Take a snapshot after rebalancing
        add_portfolio_snapshot()
        
        # Flash results
        flash(f"Rebalanceamento executado: {len(result['sells_executed'])} vendas, {len(result['buys_executed'])} compras", 'success')
        
        if result['errors']:
            for error in result['errors']:
                flash(f"Erro: {error.get('ticker', '')} - {error.get('error', '')}", 'warning')
        
        return redirect(url_for('index'))
    
    except Exception as e:
        app.logger.error(f"Erro ao executar rebalanceamento: {e}")
        app.logger.error(traceback.format_exc())
        flash(f'Erro ao executar rebalanceamento: {str(e)}', 'danger')
        return redirect(url_for('analyze'))

@app.route('/backtest', methods=['GET', 'POST'])
def backtest():
    form = BacktestForm()
    results = None
    chart_data = None
    
    if request.method == 'POST' and form.validate_on_submit():
        try:
            # Load portfolio history for backtest
            history = load_portfolio_history()
            
            if len(history) < 2:
                flash('Histórico de portfólio insuficiente para backtest. Adicione mais dados.', 'warning')
            else:
                # Run backtest
                backtest_results = create_backtest(
                    history, form.benchmark.data, form.days.data
                )
                
                if 'error' in backtest_results:
                    flash(f"Erro no backtest: {backtest_results['error']}", 'danger')
                else:
                    results = backtest_results
                    
                    # Prepare chart data
                    dates = [h['date'] for h in history[-form.days.data:] if 'date' in h]
                    portfolio_values = [h['total_value'] for h in history[-form.days.data:] if 'total_value' in h]
                    
                    # Get benchmark data for same period
                    benchmark_data = None
                    try:
                        benchmark_df = DataLoader.get_asset_data(form.benchmark.data, days=form.days.data + 5)
                        benchmark_data = benchmark_df['Close'].tolist()
                        
                        # Normalize to match portfolio starting point
                        if benchmark_data and portfolio_values:
                            scale_factor = portfolio_values[0] / benchmark_data[0]
                            benchmark_data = [price * scale_factor for price in benchmark_data]
                    except Exception as e:
                        app.logger.error(f"Error fetching benchmark data: {e}")
                    
                    chart_data = {
                        'dates': dates,
                        'portfolio': portfolio_values,
                        'benchmark': benchmark_data
                    }
        
        except Exception as e:
            app.logger.error(f"Error in backtest: {e}")
            app.logger.error(traceback.format_exc())
            flash(f'Erro ao executar backtest: {str(e)}', 'danger')
    
    return render_template('backtest.html', form=form, results=results, chart_data=chart_data)

@app.route('/optimize', methods=['GET', 'POST'])
def optimize():
    form = OptimizationForm()
    results = None
    
    if request.method == 'POST' and form.validate_on_submit():
        try:
            data = load_portfolio()
            portfolio = data.get('portfolio', {})
            watchlist = data.get('watchlist', {})
            account_balance = data.get('account_balance', 0)
            
            # Get market sentiment
            market_sentiment = MarketAnalysis.get_market_sentiment()
            
            # Create optimizer
            optimizer = PortfolioOptimizer(
                portfolio, 
                watchlist, 
                account_balance, 
                form.risk_profile.data, 
                market_sentiment
            )
            
            # Run optimization
            results = optimizer.optimize(
                max_positions=form.max_positions.data,
                target_cash_reserve_pct=form.cash_reserve_pct.data / 100,
                max_position_size_pct=form.max_position_size_pct.data / 100,
                min_score_threshold=form.min_score.data
            )
            
            # Generate rebalance plan
            rebalance_plan = optimizer.generate_rebalance_plan()
            results['rebalance_plan'] = rebalance_plan
            
        except Exception as e:
            app.logger.error(f"Error in optimization: {e}")
            app.logger.error(traceback.format_exc())
            flash(f'Erro na otimização: {str(e)}', 'danger')
    
    return render_template('optimize.html', form=form, results=results)

@app.route('/execute_optimization', methods=['POST'])
def execute_optimization():
    try:
        # Get optimization parameters
        max_positions = int(request.form.get('max_positions', 15))
        cash_reserve_pct = float(request.form.get('cash_reserve_pct', 10)) / 100
        risk_profile = request.form.get('risk_profile', 'medium')
        
        # Load portfolio data
        data = load_portfolio()
        portfolio = data.get('portfolio', {})
        watchlist = data.get('watchlist', {})
        account_balance = data.get('account_balance', 0)
        
        # Get market sentiment
        market_sentiment = MarketAnalysis.get_market_sentiment()
        
        # Create optimizer and run optimization
        optimizer = PortfolioOptimizer(
            portfolio, watchlist, account_balance, risk_profile, market_sentiment
        )
        
        results = optimizer.optimize(
            max_positions=max_positions,
            target_cash_reserve_pct=cash_reserve_pct
        )
        
        # Generate and execute rebalance plan
        rebalance_plan = optimizer.generate_rebalance_plan()
        
        # Create executor
        executor = PortfolioExecutor(portfolio, account_balance)
        
        # Execute plan
        execution_result = executor.execute_rebalance_plan(rebalance_plan)
        
        # Update portfolio data
        data['portfolio'] = execution_result['updated_portfolio']['positions']
        data['account_balance'] = execution_result['ending_balance']
        save_portfolio(data)
        
        # Take a snapshot after optimization
        add_portfolio_snapshot()
        
        # Flash results
        flash(f"Otimização executada: {len(execution_result['sells_executed'])} vendas, {len(execution_result['buys_executed'])} compras", 'success')
        
        if execution_result['errors']:
            for error in execution_result['errors']:
                flash(f"Erro: {error.get('ticker', '')} - {error.get('error', '')}", 'warning')
        
        return redirect(url_for('index'))
        
    except Exception as e:
        app.logger.error(f"Error executing optimization: {e}")
        app.logger.error(traceback.format_exc())
        flash(f'Erro ao executar otimização: {str(e)}', 'danger')
        return redirect(url_for('optimize'))

@app.route('/notifications', methods=['GET', 'POST'])
def notifications():
    form = NotificationForm()
    
    # Load notification config
    config = get_notification_config()
    
    if request.method == 'POST' and form.validate_on_submit():
        # Update notification config
        new_config = {
            "enabled": form.enabled.data,
            "methods": {
                "email": {
                    "enabled": form.email_enabled.data,
                    "address": form.email_address.data
                },
                "webhook": {
                    "enabled": form.webhook_enabled.data,
                    "url": form.webhook_url.data
                }
            },
            "notify_trades": form.notify_trades.data,
            "notify_thresholds": form.notify_thresholds.data
        }
        
        # Save config
        try:
            with open(NOTIFICATION_CONFIG, 'w') as f:
                json.dump(new_config, f, indent=2)
            
            # Update notification system
            global notification_system
            notification_system = NotificationSystem(new_config)
            
            flash('Configurações de notificação atualizadas.', 'success')
        except Exception as e:
            app.logger.error(f"Error saving notification config: {e}")
            flash(f'Erro ao salvar configurações: {str(e)}', 'danger')
        
        return redirect(url_for('notifications'))
    
    # Fill form with current values
    if request.method == 'GET':
        form.enabled.data = config.get('enabled', False)
        
        email_config = config.get('methods', {}).get('email', {})
        form.email_enabled.data = email_config.get('enabled', False)
        form.email_address.data = email_config.get('address', '')
        
        webhook_config = config.get('methods', {}).get('webhook', {})
        form.webhook_enabled.data = webhook_config.get('enabled', False)
        form.webhook_url.data = webhook_config.get('url', '')
        
        form.notify_trades.data = config.get('notify_trades', True)
        form.notify_thresholds.data = config.get('notify_thresholds', True)
    
    return render_template('notifications.html', form=form, config=config)

@app.route('/indicators', methods=['GET', 'POST'])
def indicators():
    form = IndicatorAnalysisForm()
    analysis_results = None
    charts = {}
    
    if request.method == 'POST' and form.validate_on_submit():
        ticker = form.ticker.data.upper()
        days = form.days.data
        interval = form.interval.data
        
        # Check if ticker is valid
        if not DataLoader.check_ticker_valid(ticker):
            flash(f'Ticker {ticker} inválido ou não encontrado!', 'danger')
            return render_template('indicators.html', form=form)
        
        try:
            # Get data
            df = DataLoader.get_asset_data(ticker, days, interval)
            
            if df.empty or len(df) < 5:
                flash(f'Dados insuficientes para {ticker}.', 'warning')
                return render_template('indicators.html', form=form)
            
            # Add indicators
            df_indicators = TechnicalIndicators.add_all_indicators(df, DYNAMIC_PARAMS)
            
            # Calculate some statistics
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
            
            # Generate charts
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
                ax.set_ylabel('MACD')
                ax.legend()
                ax.grid(True)
                
                # Save to buffer
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png')
                buf.seek(0)
                charts['macd'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)
            
            # Get ML forecast
            ml_forecast = trading_system.ml_price_forecast(df)
            
            analysis_results = {
                'ticker': ticker,
                'stats': stats,
                'indicators': {
                    'rsi': float(df_indicators['RSI'].iloc[-1]) if 'RSI' in df_indicators.columns else None,
                    'macd_line': float(df_indicators['MACD_line'].iloc[-1]) if 'MACD_line' in df_indicators.columns else None,
                    'macd_signal': float(df_indicators['MACD_signal'].iloc[-1]) if 'MACD_signal' in df_indicators.columns else None,
                    'macd_hist': float(df_indicators['MACD_hist'].iloc[-1]) if 'MACD_hist' in df_indicators.columns else None,
                    'bb_upper': float(df_indicators['BB_upper'].iloc[-1]) if 'BB_upper' in df_indicators.columns else None,
                    'bb_lower': float(df_indicators['BB_lower'].iloc[-1]) if 'BB_lower' in df_indicators.columns else None,
                    'adx': float(df_indicators['ADX'].iloc[-1]) if 'ADX' in df_indicators.columns else None,
                    'stoch_k': float(df_indicators['%K'].iloc[-1]) if '%K' in df_indicators.columns else None,
                    'stoch_d': float(df_indicators['%D'].iloc[-1]) if '%D' in df_indicators.columns else None,
                },
                'ml_forecast': ml_forecast
            }
            
        except Exception as e:
            app.logger.error(f"Error analyzing indicators: {e}")
            app.logger.error(traceback.format_exc())
            flash(f'Erro na análise de indicadores: {str(e)}', 'danger')
    
    return render_template('indicators.html', form=form, results=analysis_results, charts=charts)

@app.route('/fundamentals', methods=['GET', 'POST'])
def fundamentals():
    form = FundamentalAnalysisForm()
    results = None
    
    if request.method == 'POST' and form.validate_on_submit():
        ticker = form.ticker.data.upper()
        
        try:
            # Get fundamental analysis
            fundamental_results = FundamentalAnalysis.fundamental_score(ticker)
            
            if not fundamental_results or fundamental_results.get('fundamental_score', 0) == 0 and 'error' in fundamental_results.get('details', {}):
                flash(f'Não foi possível obter dados fundamentais para {ticker}.', 'warning')
                return render_template('fundamentals.html', form=form)
            
            results = fundamental_results
            
        except Exception as e:
            app.logger.error(f"Error in fundamental analysis: {e}")
            app.logger.error(traceback.format_exc())
            flash(f'Erro na análise fundamental: {str(e)}', 'danger')
    
    return render_template('fundamentals.html', form=form, results=results)

@app.route('/news', methods=['GET', 'POST'])
def news():
    form = NewsAnalysisForm()
    results = None
    
    if request.method == 'POST' and form.validate_on_submit():
        ticker = form.ticker.data.upper()
        
        try:
            # Get news sentiment analysis
            news_results = QualitativeAnalysis.analyze_news_sentiment(ticker)
            
            if not news_results or news_results.get('news_count', 0) == 0:
                flash(f'Não foram encontradas notícias para {ticker}.', 'warning')
                return render_template('news.html', form=form)
            
            # Get related tickers if API key available
            related_tickers = []
            if app.config['POLYGON_API_KEY']:
                related_tickers = APIClient.get_polygon_related(ticker)
            
            results = {
                'ticker': ticker,
                'sentiment': news_results,
                'related_tickers': related_tickers
            }
            
        except Exception as e:
            app.logger.error(f"Error in news analysis: {e}")
            app.logger.error(traceback.format_exc())
            flash(f'Erro na análise de notícias: {str(e)}', 'danger')
    
    return render_template('news.html', form=form, results=results)

@app.route('/market_sentiment')
def market_sentiment():
    try:
        sentiment = MarketAnalysis.get_market_sentiment()
        return render_template('market_sentiment.html', sentiment=sentiment)
    except Exception as e:
        app.logger.error(f"Error getting market sentiment: {e}")
        flash(f'Erro ao obter sentimento de mercado: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/parameters', methods=['GET', 'POST'])
def parameters():
    form = TradingParamsForm()
    
    if request.method == 'POST' and form.validate_on_submit():
        # Update trading parameters
        DYNAMIC_PARAMS['sma_period'] = form.sma_period.data
        DYNAMIC_PARAMS['ema_period'] = form.ema_period.data
        DYNAMIC_PARAMS['rsi_period'] = form.rsi_period.data
        DYNAMIC_PARAMS['macd_fast'] = form.macd_fast.data
        DYNAMIC_PARAMS['macd_slow'] = form.macd_slow.data
        DYNAMIC_PARAMS['macd_signal'] = form.macd_signal.data
        DYNAMIC_PARAMS['bb_window'] = form.bb_window.data
        DYNAMIC_PARAMS['bb_std'] = form.bb_std.data
        DYNAMIC_PARAMS['decision_buy_threshold'] = form.decision_buy_threshold.data
        DYNAMIC_PARAMS['decision_sell_threshold'] = form.decision_sell_threshold.data
        DYNAMIC_PARAMS['take_profit_pct'] = form.take_profit_pct.data
        DYNAMIC_PARAMS['stop_loss_pct'] = form.stop_loss_pct.data
        DYNAMIC_PARAMS['trailing_stop_pct'] = form.trailing_stop_pct.data
        
        # Save parameters to file
        try:
            params_file = os.path.join(app.config['DATA_FOLDER'], 'trading_params.json')
            with open(params_file, 'w') as f:
                json.dump(DYNAMIC_PARAMS, f, indent=2)
            flash('Parâmetros de trading atualizados com sucesso!', 'success')
        except Exception as e:
            app.logger.error(f"Error saving trading parameters: {e}")
            flash(f'Erro ao salvar parâmetros: {str(e)}', 'danger')
        
        return redirect(url_for('parameters'))
    
    # Fill form with current values
    if request.method == 'GET':
        form.sma_period.data = DYNAMIC_PARAMS.get('sma_period', 20)
        form.ema_period.data = DYNAMIC_PARAMS.get('ema_period', 9)
        form.rsi_period.data = DYNAMIC_PARAMS.get('rsi_period', 14)
        form.macd_fast.data = DYNAMIC_PARAMS.get('macd_fast', 12)
        form.macd_slow.data = DYNAMIC_PARAMS.get('macd_slow', 26)
        form.macd_signal.data = DYNAMIC_PARAMS.get('macd_signal', 9)
        form.bb_window.data = DYNAMIC_PARAMS.get('bb_window', 20)
        form.bb_std.data = DYNAMIC_PARAMS.get('bb_std', 2)
        form.decision_buy_threshold.data = DYNAMIC_PARAMS.get('decision_buy_threshold', 60)
        form.decision_sell_threshold.data = DYNAMIC_PARAMS.get('decision_sell_threshold', -60)
        form.take_profit_pct.data = DYNAMIC_PARAMS.get('take_profit_pct', 5.0)
        form.stop_loss_pct.data = DYNAMIC_PARAMS.get('stop_loss_pct', -8.0)
        form.trailing_stop_pct.data = DYNAMIC_PARAMS.get('trailing_stop_pct', 3.0)
    
    return render_template('parameters.html', form=form, params=DYNAMIC_PARAMS)

@app.route('/diagnostics')
def diagnostics():
    try:
        # Run system diagnostics
        results = run_diagnostics()
        
        # Check for updates
        update_info = check_for_updates(VERSION)
        
        return render_template('diagnostics.html', results=results, update_info=update_info, version=VERSION)
    except Exception as e:
        app.logger.error(f"Error running diagnostics: {e}")
        flash(f'Erro ao executar diagnósticos: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/download_portfolio')
def download_portfolio():
    try:
        data = load_portfolio()
        
        # Generate in-memory file
        memory_file = BytesIO()
        memory_file.write(json.dumps(data, indent=2).encode('utf-8'))
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            as_attachment=True,
            download_name='portfolio_export.json',
            mimetype='application/json'
        )
    except Exception as e:
        app.logger.error(f"Error downloading portfolio: {e}")
        flash(f'Erro ao baixar portfólio: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/upload-json', methods=['GET', 'POST'])
def upload_json():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Nenhum arquivo selecionado', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('Nenhum arquivo selecionado', 'danger')
            return redirect(request.url)
        
        if file:
            try:
                data = json.load(file)
                save_portfolio(data)
                flash('Portfólio carregado com sucesso!', 'success')
                return redirect(url_for('index'))
            except Exception as e:
                flash(f'Erro ao carregar arquivo: {str(e)}', 'danger')
    
    return render_template('upload.html')

# API Routes
@app.route('/api/portfolio')
def api_portfolio():
    data = load_portfolio()
    return jsonify(data)

@app.route('/api/market_sentiment')
def api_market_sentiment():
    try:
        sentiment = MarketAnalysis.get_market_sentiment()
        return jsonify(sentiment)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        data = request.json
        portfolio = data.get('portfolio', {})
        account_balance = data.get('account_balance', 0)
        risk_profile = data.get('risk_profile', 'medium')
        
        portfolio_analysis = analyze_portfolio(
            portfolio, account_balance, risk_profile, {}, False, {}
        )
        
        return jsonify(portfolio_analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ticker_data/<ticker>')
def api_ticker_data(ticker):
    try:
        days = int(request.args.get('days', 60))
        interval = request.args.get('interval', '1d')
        
        df = DataLoader.get_asset_data(ticker, days, interval)
        
        if df.empty:
            return jsonify({"error": "No data available"}), 404
        
        # Convert to JSON-friendly format
        df_json = df.reset_index()
        if 'Date' in df_json.columns:
            df_json['Date'] = df_json['Date'].dt.strftime('%Y-%m-%d')
        
        data = df_json.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize app with parameters from file
def init_app():
    # Load trading parameters if available
    params_file = os.path.join(app.config['DATA_FOLDER'], 'trading_params.json')
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
                # Update global params
                for key, value in params.items():
                    DYNAMIC_PARAMS[key] = value
        except Exception as e:
            app.logger.error(f"Error loading trading parameters: {e}")
    
    # Set Polygon API key if available
    if app.config['POLYGON_API_KEY']:
        trading_system.POLYGON_API_KEY = app.config['POLYGON_API_KEY']

# Initialize
init_app()

# Iniciar o servidor se executado diretamente
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)