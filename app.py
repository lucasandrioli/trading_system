import os
import json
import importlib
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, FloatField, IntegerField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Optional, NumberRange

# Importar o sistema de trading existente
import trading_system
from trading_system import (
    DataLoader, TechnicalIndicators, MarketAnalysis, Strategy,
    FundamentalAnalysis, QualitativeAnalysis, analyze_portfolio,
    analyze_watchlist, generate_rebalance_plan, DYNAMIC_PARAMS
)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'trading-system-secret-key')
app.config['DATA_FOLDER'] = os.environ.get('DATA_FOLDER', 'data')
csrf = CSRFProtect(app)
CORS(app)

# Garantir que o diretório de dados existe
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
PORTFOLIO_FILE = os.path.join(app.config['DATA_FOLDER'], 'portfolio.json')

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
    
    summary = {
        'total_invested': total_invested,
        'portfolio_value': portfolio_value,
        'account_balance': account_balance,
        'total_value': total_value,
        'profit_loss': profit_loss,
        'profit_loss_pct': profit_loss_pct,
        'position_count': len(portfolio)
    }
    
    goals = data.get('goals', {})
    
    return render_template(
        'index.html', 
        portfolio=portfolio, 
        watchlist=data.get('watchlist', {}),
        summary=summary,
        goals=goals
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
            
            # Adicionar ou atualizar ativo
            if ticker in data['portfolio']:
                # Cálculo de preço médio ponderado
                old_qty = data['portfolio'][ticker].get('quantity', 0)
                old_price = data['portfolio'][ticker].get('avg_price', 0)
                new_qty = form.quantity.data
                new_price = form.avg_price.data
                
                total_qty = old_qty + new_qty
                avg_price = ((old_qty * old_price) + (new_qty * new_price)) / total_qty if total_qty > 0 else 0
                
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
        else:
            flash(f'Ticker {ticker} inválido ou não encontrado!', 'danger')
        
    return redirect(url_for('portfolio'))

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
        data['account_balance'] = form.balance.data
        save_portfolio(data)
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
        flash('Metas atualizadas com sucesso!', 'success')
        return redirect(url_for('goals'))
    
    if data.get('goals'):
        form.target_recovery.data = data['goals'].get('target_recovery', 0)
        form.days.data = data['goals'].get('days', 0)
    
    return render_template('goals.html', form=form, goals=data.get('goals', {}))

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
            # Verificar saldo
            if operation_value + commission > data.get('account_balance', 0):
                flash('Saldo insuficiente para esta operação!', 'danger')
                return redirect(url_for('trade'))
            
            # Atualizar portfólio
            if ticker in data['portfolio']:
                # Cálculo de preço médio ponderado
                old_qty = data['portfolio'][ticker].get('quantity', 0)
                old_price = data['portfolio'][ticker].get('avg_price', 0)
                
                total_qty = old_qty + quantity
                avg_price = ((old_qty * old_price) + (quantity * price)) / total_qty if total_qty > 0 else 0
                
                data['portfolio'][ticker]['quantity'] = total_qty
                data['portfolio'][ticker]['avg_price'] = avg_price
                data['portfolio'][ticker]['current_price'] = price
                data['portfolio'][ticker]['current_position'] = total_qty * price
                data['portfolio'][ticker]['last_buy'] = {
                    'price': price,
                    'quantity': quantity,
                    'date': datetime.now().isoformat()
                }
            else:
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
        
        elif trade_type == "VENDA":
            # Verificar se possui o ativo e a quantidade
            if ticker not in data['portfolio']:
                flash(f'Você não possui {ticker} em sua carteira!', 'danger')
                return redirect(url_for('trade'))
            
            current_qty = data['portfolio'][ticker].get('quantity', 0)
            if quantity > current_qty:
                flash(f'Quantidade insuficiente. Você possui {current_qty} {ticker}.', 'danger')
                return redirect(url_for('trade'))
            
            # Atualizar portfólio
            data['portfolio'][ticker]['quantity'] -= quantity
            data['portfolio'][ticker]['current_price'] = price
            data['portfolio'][ticker]['current_position'] = data['portfolio'][ticker]['quantity'] * price
            data['portfolio'][ticker]['last_sell'] = {
                'price': price,
                'quantity': quantity,
                'date': datetime.now().isoformat()
            }
            
            # Se quantidade = 0, remover ativo
            if data['portfolio'][ticker]['quantity'] <= 0:
                del data['portfolio'][ticker]
            
            # Atualizar saldo
            data['account_balance'] += (operation_value - commission)
            flash(f'Venda de {quantity} {ticker} a ${price:.2f} registrada com sucesso!', 'success')
        
        save_portfolio(data)
        return redirect(url_for('index'))
    
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
        trailing_data = {}
        extended_hours = False
        
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
        
        # Guardar análise para uso na página
        return render_template(
            'analysis.html',
            portfolio_analysis=portfolio_analysis,
            watchlist_analysis=watchlist_analysis,
            rebalance_plan=rebalance_plan,
            risk_profile=risk_profile
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
        
        # Aqui você implementaria a lógica para aplicar automaticamente
        # a transação de rebalanceamento com base no ID e ação
        
        flash('Funcionalidade em desenvolvimento. Por favor, aplique manualmente.', 'warning')
        return redirect(url_for('analyze'))
    
    except Exception as e:
        app.logger.error(f"Erro ao aplicar rebalanceamento: {e}")
        flash(f'Erro ao aplicar rebalanceamento: {str(e)}', 'danger')
        return redirect(url_for('analyze'))

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

# Iniciar o servidor se executado diretamente
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)