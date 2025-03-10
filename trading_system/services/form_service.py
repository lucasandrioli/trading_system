from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, IntegerField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired, Optional, NumberRange

class FormService:
    """Service for creating and validating forms."""
    
    @staticmethod
    def create_asset_form(ticker=None, quantity=None, avg_price=None):
        """Create a form for adding/editing portfolio assets."""
        class AssetForm(FlaskForm):
            ticker = StringField('Ticker', validators=[DataRequired()])
            quantity = IntegerField('Quantidade', validators=[DataRequired(), NumberRange(min=1)])
            avg_price = FloatField('Preço Médio', validators=[DataRequired(), NumberRange(min=0)])
            submit = SubmitField('Salvar')
        
        form = AssetForm()
        
        # Pre-fill form data if provided
        if ticker:
            form.ticker.data = ticker
        if quantity:
            form.quantity.data = quantity
        if avg_price:
            form.avg_price.data = avg_price
            
        return form
    
    @staticmethod
    def create_watchlist_form(ticker=None, monitor=True):
        """Create a form for adding to watchlist."""
        class WatchlistForm(FlaskForm):
            ticker = StringField('Ticker', validators=[DataRequired()])
            monitor = BooleanField('Monitorar', default=True)
            submit = SubmitField('Adicionar')
        
        form = WatchlistForm()
        
        if ticker:
            form.ticker.data = ticker
        form.monitor.data = monitor
        
        return form
    
    @staticmethod
    def create_account_form(balance=0.0):
        """Create a form for updating account balance."""
        class AccountForm(FlaskForm):
            balance = FloatField('Saldo da Conta', validators=[DataRequired(), NumberRange(min=0)])
            submit = SubmitField('Atualizar')
        
        form = AccountForm()
        form.balance.data = balance
        return form
    
    @staticmethod
    def create_goals_form(target=None, days=None):
        """Create a form for setting recovery goals."""
        class GoalsForm(FlaskForm):
            target_recovery = FloatField('Meta de Recuperação ($)', 
                                       validators=[Optional(), NumberRange(min=0)])
            days = IntegerField('Prazo (dias)', 
                               validators=[Optional(), NumberRange(min=1)])
            submit = SubmitField('Definir Metas')
        
        form = GoalsForm()
        
        if target:
            form.target_recovery.data = target
        if days:
            form.days.data = days
            
        return form
    
    @staticmethod
    def create_trade_form(ticker=None, quantity=None, price=None, trade_type=None):
        """Create a form for trading."""
        class TradeForm(FlaskForm):
            ticker = StringField('Ticker', validators=[DataRequired()])
            quantity = IntegerField('Quantidade', validators=[DataRequired(), NumberRange(min=1)])
            price = FloatField('Preço', validators=[DataRequired(), NumberRange(min=0.01)])
            trade_type = StringField('Tipo (COMPRA/VENDA)', validators=[DataRequired()])
            submit = SubmitField('Registrar')
        
        form = TradeForm()
        
        if ticker:
            form.ticker.data = ticker
        if quantity:
            form.quantity.data = quantity
        if price:
            form.price.data = price
        if trade_type:
            form.trade_type.data = trade_type
            
        return form
    
    @staticmethod
    def create_backtest_form(days=30, benchmark="SPY", risk_profile="medium"):
        """Create a form for backtesting."""
        class BacktestForm(FlaskForm):
            days = IntegerField('Período (dias)', 
                              validators=[DataRequired(), NumberRange(min=5, max=365)])
            benchmark = StringField('Benchmark', validators=[DataRequired()])
            risk_profile = SelectField('Perfil de Risco', choices=[
                ('low', 'Baixo'), 
                ('medium', 'Médio'), 
                ('high', 'Alto'), 
                ('ultra', 'Ultra')
            ])
            submit = SubmitField('Executar Backtest')
        
        form = BacktestForm()
        form.days.data = days
        form.benchmark.data = benchmark
        form.risk_profile.data = risk_profile
        
        return form
    
    @staticmethod
    def create_optimization_form(max_positions=15, cash_reserve_pct=10, max_position_size_pct=20, 
                               min_score=30, risk_profile="medium"):
        """Create a form for portfolio optimization."""
        class OptimizationForm(FlaskForm):
            max_positions = IntegerField('Máximo de Posições', 
                                       validators=[DataRequired(), NumberRange(min=1, max=50)])
            cash_reserve_pct = FloatField('Reserva de Caixa (%)', 
                                        validators=[DataRequired(), NumberRange(min=0, max=50)])
            max_position_size_pct = FloatField('Tamanho Máximo de Posição (%)', 
                                             validators=[DataRequired(), NumberRange(min=1, max=50)])
            min_score = FloatField('Score Mínimo para Inclusão', 
                                 validators=[DataRequired(), NumberRange(min=0, max=100)])
            risk_profile = SelectField('Perfil de Risco', choices=[
                ('low', 'Baixo'), 
                ('medium', 'Médio'), 
                ('high', 'Alto'), 
                ('ultra', 'Ultra')
            ])
            submit = SubmitField('Otimizar Carteira')
        
        form = OptimizationForm()
        form.max_positions.data = max_positions
        form.cash_reserve_pct.data = cash_reserve_pct
        form.max_position_size_pct.data = max_position_size_pct
        form.min_score.data = min_score
        form.risk_profile.data = risk_profile
        
        return form
    
    @staticmethod
    def create_notification_form(enabled=False, email_enabled=False, email_address="",
                               webhook_enabled=False, webhook_url="", notify_trades=True,
                               notify_thresholds=True):
        """Create a form for notification settings."""
        class NotificationForm(FlaskForm):
            enabled = BooleanField('Ativar Notificações')
            email_enabled = BooleanField('Notificações por Email')
            email_address = StringField('Endereço de Email')
            webhook_enabled = BooleanField('Webhook (Discord/Slack)')
            webhook_url = StringField('URL do Webhook')
            notify_trades = BooleanField('Notificar Operações')
            notify_thresholds = BooleanField('Notificar Limiares de Preço')
            submit = SubmitField('Salvar Configurações')
        
        form = NotificationForm()
        form.enabled.data = enabled
        form.email_enabled.data = email_enabled
        form.email_address.data = email_address
        form.webhook_enabled.data = webhook_enabled
        form.webhook_url.data = webhook_url
        form.notify_trades.data = notify_trades
        form.notify_thresholds.data = notify_thresholds
        
        return form
    
    @staticmethod
    def create_indicator_analysis_form(ticker="", days=60, interval="1d"):
        """Create a form for technical indicator analysis."""
        class IndicatorAnalysisForm(FlaskForm):
            ticker = StringField('Ticker', validators=[DataRequired()])
            days = IntegerField('Período (dias)', 
                              validators=[DataRequired(), NumberRange(min=5, max=365)])
            interval = SelectField('Intervalo', choices=[
                ('1d', 'Diário'),
                ('1h', 'Horário'),
                ('1wk', 'Semanal')
            ])
            submit = SubmitField('Analisar')
        
        form = IndicatorAnalysisForm()
        form.ticker.data = ticker
        form.days.data = days
        form.interval.data = interval
        
        return form
    
    @staticmethod
    def create_fundamental_analysis_form(ticker=""):
        """Create a form for fundamental analysis."""
        class FundamentalAnalysisForm(FlaskForm):
            ticker = StringField('Ticker', validators=[DataRequired()])
            submit = SubmitField('Analisar')
        
        form = FundamentalAnalysisForm()
        form.ticker.data = ticker
        
        return form
    
    @staticmethod
    def create_news_analysis_form(ticker=""):
        """Create a form for news analysis."""
        class NewsAnalysisForm(FlaskForm):
            ticker = StringField('Ticker', validators=[DataRequired()])
            submit = SubmitField('Buscar Notícias')
        
        form = NewsAnalysisForm()
        form.ticker.data = ticker
        
        return form
    
    @staticmethod
    def create_parameters_form(params=None):
        """Create a form for trading parameters."""
        if params is None:
            params = {}
            
        class TradingParamsForm(FlaskForm):
            sma_period = IntegerField('SMA Period', 
                                    validators=[NumberRange(min=1, max=200)])
            ema_period = IntegerField('EMA Period', 
                                    validators=[NumberRange(min=1, max=200)])
            rsi_period = IntegerField('RSI Period', 
                                    validators=[NumberRange(min=1, max=50)])
            macd_fast = IntegerField('MACD Fast', 
                                   validators=[NumberRange(min=1, max=50)])
            macd_slow = IntegerField('MACD Slow', 
                                   validators=[NumberRange(min=1, max=50)])
            macd_signal = IntegerField('MACD Signal', 
                                     validators=[NumberRange(min=1, max=50)])
            bb_window = IntegerField('Bollinger Window', 
                                   validators=[NumberRange(min=1, max=100)])
            bb_std = FloatField('Bollinger Std', 
                              validators=[NumberRange(min=0.5, max=4)])
            decision_buy_threshold = FloatField('Buy Threshold', 
                                             validators=[NumberRange(min=0, max=100)])
            decision_sell_threshold = FloatField('Sell Threshold', 
                                              validators=[NumberRange(min=-100, max=0)])
            take_profit_pct = FloatField('Take Profit %', 
                                       validators=[NumberRange(min=0.5, max=50)])
            stop_loss_pct = FloatField('Stop Loss %', 
                                     validators=[NumberRange(min=-50, max=-0.5)])
            trailing_stop_pct = FloatField('Trailing Stop %', 
                                         validators=[NumberRange(min=0.5, max=20)])
            submit = SubmitField('Atualizar Parâmetros')
        
        form = TradingParamsForm()
        
        # Fill form with current values
        form.sma_period.data = params.get('sma_period', 20)
        form.ema_period.data = params.get('ema_period', 9)
        form.rsi_period.data = params.get('rsi_period', 14)
        form.macd_fast.data = params.get('macd_fast', 12)
        form.macd_slow.data = params.get('macd_slow', 26)
        form.macd_signal.data = params.get('macd_signal', 9)
        form.bb_window.data = params.get('bb_window', 20)
        form.bb_std.data = params.get('bb_std', 2)
        form.decision_buy_threshold.data = params.get('decision_buy_threshold', 60)
        form.decision_sell_threshold.data = params.get('decision_sell_threshold', -60)
        form.take_profit_pct.data = params.get('take_profit_pct', 5.0)
        form.stop_loss_pct.data = params.get('stop_loss_pct', -8.0)
        form.trailing_stop_pct.data = params.get('trailing_stop_pct', 3.0)
        
        return form