import logging
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, send_file, jsonify
from datetime import datetime
from io import BytesIO
import json
import base64

# Configure logging
logger = logging.getLogger("trading_system.web.routes")

# Create web blueprint
web_blueprint = Blueprint('web', __name__)

@web_blueprint.route('/')
def index():
    """Homepage with portfolio dashboard."""
    try:
        # Get portfolio data
        portfolio_service = current_app.portfolio_service
        data = portfolio_service.load_portfolio()
        
        # Calculate portfolio totals
        portfolio = data.get('portfolio', {})
        account_balance = data.get('account_balance', 0)
        goals = data.get('goals', {})
        
        # Get market sentiment
        market_service = current_app.market_service
        market_sentiment = market_service.get_market_sentiment()
        
        # Get portfolio summary
        summary = portfolio_service.get_portfolio_summary(portfolio, account_balance, market_sentiment, goals)
        
        # Check for updates
        system_service = current_app.system_service
        update_info = system_service.check_for_updates()
        
        # Take a daily snapshot if we don't have one for today
        history_service = current_app.history_service
        last_history = history_service.load_portfolio_history()
        if not last_history or last_history[-1]['date'] != datetime.now().strftime("%Y-%m-%d"):
            history_service.add_portfolio_snapshot(portfolio, account_balance)
        
        return render_template(
            'index.html', 
            portfolio=portfolio, 
            watchlist=data.get('watchlist', {}),
            summary=summary,
            goals=goals,
            update_info=update_info,
            version=system_service.VERSION
        )
    except Exception as e:
        logger.error(f"Error in index route: {e}", exc_info=True)
        flash("An error occurred while loading the dashboard. Please try again.", "danger")
        return render_template('error.html', error=str(e))

@web_blueprint.route('/portfolio')
def portfolio():
    """Portfolio management page."""
    try:
        portfolio_service = current_app.portfolio_service
        data = portfolio_service.load_portfolio()
        form_service = current_app.form_service
        form = form_service.get_asset_form()
        return render_template('portfolio.html', portfolio=data.get('portfolio', {}), form=form)
    except Exception as e:
        logger.error(f"Error in portfolio route: {e}", exc_info=True)
        flash("An error occurred while loading the portfolio. Please try again.", "danger")
        return redirect(url_for('web.index'))

@web_blueprint.route('/portfolio/add', methods=['POST'])
def add_asset():
    """Add or update an asset in the portfolio."""
    form_service = current_app.form_service
    form = form_service.get_asset_form()
    
    if form.validate_on_submit():
        portfolio_service = current_app.portfolio_service
        ticker = form.ticker.data.upper()
        
        result = portfolio_service.add_asset_to_portfolio(
            ticker,
            form.quantity.data,
            form.avg_price.data
        )
        
        if result.get('success'):
            flash(result.get('message', 'Asset added/updated successfully!'), 'success')
            
            # Take a snapshot after portfolio change
            history_service = current_app.history_service
            history_service.add_portfolio_snapshot()
            
            # Send notification
            notification_service = current_app.notification_service
            notification_service.notify_trade(
                ticker, 
                "COMPRA" if result.get('is_new') else "ATUALIZAÇÃO", 
                form.quantity.data, 
                form.avg_price.data, 
                "Adicionado/atualizado manualmente à carteira"
            )
        else:
            flash(result.get('message', 'Error adding asset to portfolio.'), 'danger')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"{getattr(form, field).label.text}: {error}", "danger")
    
    return redirect(url_for('web.portfolio'))

@web_blueprint.route('/portfolio/edit/<ticker>', methods=['GET', 'POST'])
def edit_asset(ticker):
    """Edit an existing asset in the portfolio."""
    portfolio_service = current_app.portfolio_service
    form_service = current_app.form_service
    data = portfolio_service.load_portfolio()
    
    if ticker not in data['portfolio']:
        flash(f'Asset {ticker} not found in portfolio!', 'danger')
        return redirect(url_for('web.portfolio'))
    
    form = form_service.get_asset_form()
    
    if request.method == 'POST' and form.validate_on_submit():
        result = portfolio_service.update_asset(
            ticker,
            form.quantity.data,
            form.avg_price.data
        )
        
        if result.get('success'):
            flash(result.get('message', f'Asset {ticker} updated successfully!'), 'success')
            
            # Take portfolio snapshot
            history_service = current_app.history_service
            history_service.add_portfolio_snapshot()
            
            return redirect(url_for('web.portfolio'))
        else:
            flash(result.get('message', f'Error updating asset {ticker}.'), 'danger')
    
    # Populate form with existing data
    if request.method == 'GET':
        position = data['portfolio'][ticker]
        form.ticker.data = ticker
        form.quantity.data = position.get('quantity', 0)
        form.avg_price.data = position.get('avg_price', 0)
    
    return render_template('edit_asset.html', form=form, ticker=ticker, asset=data['portfolio'][ticker])

@web_blueprint.route('/portfolio/delete/<ticker>')
def delete_asset(ticker):
    """Remove an asset from the portfolio."""
    try:
        portfolio_service = current_app.portfolio_service
        result = portfolio_service.delete_asset(ticker)
        
        if result.get('success'):
            flash(result.get('message', f'Asset {ticker} removed successfully!'), 'success')
        else:
            flash(result.get('message', f'Asset {ticker} not found!'), 'danger')
        
        return redirect(url_for('web.portfolio'))
    except Exception as e:
        logger.error(f"Error deleting asset {ticker}: {e}", exc_info=True)
        flash(f"An error occurred while removing the asset: {str(e)}", "danger")
        return redirect(url_for('web.portfolio'))

@web_blueprint.route('/watchlist')
def watchlist():
    """Watchlist management page."""
    try:
        portfolio_service = current_app.portfolio_service
        data = portfolio_service.load_portfolio()
        form_service = current_app.form_service
        form = form_service.get_watchlist_form()
        return render_template('watchlist.html', watchlist=data.get('watchlist', {}), form=form)
    except Exception as e:
        logger.error(f"Error in watchlist route: {e}", exc_info=True)
        flash("An error occurred while loading the watchlist. Please try again.", "danger")
        return redirect(url_for('web.index'))

@web_blueprint.route('/watchlist/add', methods=['POST'])
def add_to_watchlist():
    """Add a ticker to the watchlist."""
    form_service = current_app.form_service
    form = form_service.get_watchlist_form()
    
    if form.validate_on_submit():
        portfolio_service = current_app.portfolio_service
        ticker = form.ticker.data.upper()
        
        result = portfolio_service.add_to_watchlist(ticker, form.monitor.data)
        
        if result.get('success'):
            flash(result.get('message', f'Ticker {ticker} added to watchlist!'), 'success')
        else:
            flash(result.get('message', f'Ticker {ticker} invalid or not found!'), 'danger')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"{getattr(form, field).label.text}: {error}", "danger")
    
    return redirect(url_for('web.watchlist'))

@web_blueprint.route('/watchlist/delete/<ticker>')
def delete_from_watchlist(ticker):
    """Remove a ticker from the watchlist."""
    try:
        portfolio_service = current_app.portfolio_service
        result = portfolio_service.delete_from_watchlist(ticker)
        
        if result.get('success'):
            flash(result.get('message', f'Ticker {ticker} removed from watchlist!'), 'success')
        else:
            flash(result.get('message', f'Ticker {ticker} not found in watchlist!'), 'danger')
        
        return redirect(url_for('web.watchlist'))
    except Exception as e:
        logger.error(f"Error removing from watchlist {ticker}: {e}", exc_info=True)
        flash(f"An error occurred while removing from watchlist: {str(e)}", "danger")
        return redirect(url_for('web.watchlist'))

@web_blueprint.route('/account', methods=['GET', 'POST'])
def account():
    """Account balance management page."""
    portfolio_service = current_app.portfolio_service
    form_service = current_app.form_service
    data = portfolio_service.load_portfolio()
    form = form_service.get_account_form()
    
    if request.method == 'POST' and form.validate_on_submit():
        result = portfolio_service.update_account_balance(form.balance.data)
        
        if result.get('success'):
            # If significant change, log it in history
            if abs(form.balance.data - result.get('old_balance', 0)) > 1:
                history_service = current_app.history_service
                history_service.add_portfolio_snapshot()
            
            flash('Account balance updated successfully!', 'success')
            return redirect(url_for('web.account'))
        else:
            flash(result.get('message', 'Error updating account balance.'), 'danger')
    
    form.balance.data = data.get('account_balance', 0)
    return render_template('account.html', form=form, account_balance=data.get('account_balance', 0))

@web_blueprint.route('/goals', methods=['GET', 'POST'])
def goals():
    """Recovery goals management page."""
    portfolio_service = current_app.portfolio_service
    form_service = current_app.form_service
    history_service = current_app.history_service
    data = portfolio_service.load_portfolio()
    form = form_service.get_goals_form()
    
    if request.method == 'POST' and form.validate_on_submit():
        result = portfolio_service.update_goals(
            form.target_recovery.data,
            form.days.data
        )
        
        if result.get('success'):
            # Take a snapshot when goals are set
            history_service.add_portfolio_snapshot()
            flash('Goals updated successfully!', 'success')
            return redirect(url_for('web.goals'))
        else:
            flash(result.get('message', 'Error updating goals.'), 'danger')
    
    if data.get('goals'):
        form.target_recovery.data = data['goals'].get('target_recovery', 0)
        form.days.data = data['goals'].get('days', 0)
    
    # Get recovery tracker data
    recovery_data = portfolio_service.get_recovery_data(data)
    
    return render_template('goals.html', form=form, goals=data.get('goals', {}), recovery_data=recovery_data)

@web_blueprint.route('/trade', methods=['GET', 'POST'])
def trade():
    """Trade execution page."""
    portfolio_service = current_app.portfolio_service
    form_service = current_app.form_service
    notification_service = current_app.notification_service
    history_service = current_app.history_service
    data = portfolio_service.load_portfolio()
    form = form_service.get_trade_form()
    
    if request.method == 'POST' and form.validate_on_submit():
        ticker = form.ticker.data.upper()
        quantity = form.quantity.data
        price = form.price.data
        trade_type = form.trade_type.data.upper()
        
        if trade_type not in ["COMPRA", "VENDA"]:
            flash('Invalid operation type. Use COMPRA or VENDA.', 'danger')
            return redirect(url_for('web.trade'))
        
        # Execute trade
        result = portfolio_service.execute_trade(ticker, quantity, price, trade_type)
        
        if result.get('success'):
            flash(result.get('message', f'Trade for {quantity} {ticker} at ${price:.2f} registered successfully!'), 'success')
            
            # Send notification
            notification_service.notify_trade_execution({
                "action": trade_type,
                "ticker": ticker,
                "quantity": quantity,
                "price": price,
                "gross_value": result.get('operation_value', 0),
                "commission": result.get('commission', 0)
            })
            
            # Take a snapshot after trade
            history_service.add_portfolio_snapshot()
            
            return redirect(url_for('web.index'))
        else:
            flash(result.get('message', 'Error executing trade.'), 'danger')
    
    # Pre-fill form from query parameters
    if request.method == 'GET':
        form.ticker.data = request.args.get('ticker', '')
        form.quantity.data = request.args.get('quantity', None)
        form.price.data = request.args.get('price', None)
        form.trade_type.data = request.args.get('trade_type', '')
    
    return render_template('trade.html', form=form)

@web_blueprint.route('/refresh_prices')
def refresh_prices():
    """Update current prices for all portfolio assets."""
    try:
        portfolio_service = current_app.portfolio_service
        result = portfolio_service.refresh_prices()
        
        if result.get('success'):
            for update in result.get('updates', []):
                flash(update, 'info')
            flash('Prices updated successfully!', 'success')
        else:
            flash(result.get('message', 'Error updating prices.'), 'danger')
        
        return redirect(url_for('web.index'))
    except Exception as e:
        logger.error(f"Error refreshing prices: {e}", exc_info=True)
        flash(f"An error occurred while updating prices: {str(e)}", "danger")
        return redirect(url_for('web.index'))

@web_blueprint.route('/analyze')
def analyze():
    """Portfolio and watchlist analysis page."""
    try:
        # Get portfolio data
        portfolio_service = current_app.portfolio_service
        analysis_service = current_app.analysis_service
        data = portfolio_service.load_portfolio()
        portfolio = data.get('portfolio', {})
        watchlist = data.get('watchlist', {})
        account_balance = data.get('account_balance', 0)
        goals = data.get('goals', {})
        
        if not portfolio:
            flash('Add assets to your portfolio for analysis.', 'warning')
            return redirect(url_for('web.index'))
        
        # Get analysis settings
        risk_profile = request.args.get('risk', 'medium')
        extended_hours = request.args.get('extended', 'false').lower() == 'true'
        quick_mode = request.args.get('quick', 'true').lower() == 'true'
        
        logger.info(f"Starting analysis with risk profile: {risk_profile}, quick mode: {quick_mode}")
        
        # Execute optimized analyses with error handling
        try:
            portfolio_analysis = analysis_service.analyze_portfolio(
                portfolio, account_balance, risk_profile,
                extended_hours, goals, quick_mode
            )
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}", exc_info=True)
            portfolio_analysis = {"ativos": {}, "resumo": {}}
            flash("Error analyzing portfolio. Using partial results.", "warning")
        
        try:
            watchlist_analysis = analysis_service.analyze_watchlist(
                watchlist, account_balance, risk_profile,
                extended_hours, goals, quick_mode
            )
        except Exception as e:
            logger.error(f"Error in watchlist analysis: {e}", exc_info=True)
            watchlist_analysis = {}
            flash("Error analyzing watchlist. Using partial results.", "warning")
        
        try:
            rebalance_plan = analysis_service.generate_rebalance_plan(
                portfolio_analysis, watchlist_analysis,
                account_balance
            )
        except Exception as e:
            logger.error(f"Error generating rebalance plan: {e}", exc_info=True)
            rebalance_plan = {"sell": [], "buy": [], "rebalance": [], "stats": {}}
            flash("Error generating rebalance plan. Using partial results.", "warning")
        
        # Ensure resumo exists if missing and has all required fields
        analysis_service.ensure_complete_analysis_results(portfolio_analysis)
        
        return render_template(
            'analysis.html',
            portfolio_analysis=portfolio_analysis,
            watchlist_analysis=watchlist_analysis,
            rebalance_plan=rebalance_plan,
            risk_profile=risk_profile,
            extended_hours=extended_hours,
            quick_mode=quick_mode
        )
    
    except Exception as e:
        logger.error(f"Error in analysis: {e}", exc_info=True)
        flash(f'Error performing analysis: {str(e)}', 'danger')
        return redirect(url_for('web.index'))

@web_blueprint.route('/execute_rebalance', methods=['POST'])
def execute_rebalance():
    """Execute the recommended rebalancing plan."""
    try:
        portfolio_service = current_app.portfolio_service
        analysis_service = current_app.analysis_service
        history_service = current_app.history_service
        
        # Get risk profile
        risk_profile = request.form.get('risk_profile', 'medium')
        
        # Execute rebalance
        result = analysis_service.execute_rebalance_plan(risk_profile)
        
        if result.get('success'):
            # Take a snapshot after rebalancing
            history_service.add_portfolio_snapshot()
            
            # Flash results
            flash(result.get('message', "Rebalancing executed successfully"), 'success')
            
            if result.get('errors'):
                for error in result.get('errors'):
                    flash(f"Error: {error.get('ticker', '')} - {error.get('error', '')}", 'warning')
        else:
            flash(result.get('message', 'Error executing rebalance plan.'), 'danger')
        
        return redirect(url_for('web.index'))
    
    except Exception as e:
        logger.error(f"Error executing rebalance: {e}", exc_info=True)
        flash(f'Error executing rebalance plan: {str(e)}', 'danger')
        return redirect(url_for('web.analyze'))

@web_blueprint.route('/backtest', methods=['GET', 'POST'])
def backtest():
    """Portfolio backtest page."""
    form_service = current_app.form_service
    form = form_service.get_backtest_form()
    results = None
    chart_data = None
    
    if request.method == 'POST' and form.validate_on_submit():
        try:
            history_service = current_app.history_service
            analysis_service = current_app.analysis_service
            
            # Load portfolio history for backtest
            history = history_service.load_portfolio_history()
            
            if len(history) < 2:
                flash('Insufficient portfolio history for backtest. Add more data.', 'warning')
            else:
                # Run backtest
                backtest_results = analysis_service.create_backtest(
                    history, form.benchmark.data, form.days.data
                )
                
                if 'error' in backtest_results:
                    flash(f"Error in backtest: {backtest_results['error']}", 'danger')
                else:
                    results = backtest_results
                    chart_data = analysis_service.prepare_backtest_chart_data(
                        history, form.benchmark.data, form.days.data
                    )
        
        except Exception as e:
            logger.error(f"Error in backtest: {e}", exc_info=True)
            flash(f'Error executing backtest: {str(e)}', 'danger')
    
    return render_template('backtest.html', form=form, results=results, chart_data=chart_data)

@web_blueprint.route('/optimize', methods=['GET', 'POST'])
def optimize():
    """Portfolio optimization page."""
    form_service = current_app.form_service
    form = form_service.get_optimization_form()
    results = None
    
    if request.method == 'POST' and form.validate_on_submit():
        try:
            analysis_service = current_app.analysis_service
            
            # Run optimization
            results = analysis_service.optimize_portfolio(
                max_positions=form.max_positions.data,
                cash_reserve_pct=form.cash_reserve_pct.data / 100,
                max_position_size_pct=form.max_position_size_pct.data / 100,
                min_score=form.min_score.data,
                risk_profile=form.risk_profile.data
            )
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}", exc_info=True)
            flash(f'Error in optimization: {str(e)}', 'danger')
    
    return render_template('optimize.html', form=form, results=results)

@web_blueprint.route('/execute_optimization', methods=['POST'])
def execute_optimization():
    """Execute the recommended portfolio optimization."""
    try:
        # Get optimization parameters
        max_positions = int(request.form.get('max_positions', 15))
        cash_reserve_pct = float(request.form.get('cash_reserve_pct', 10)) / 100
        risk_profile = request.form.get('risk_profile', 'medium')
        
        analysis_service = current_app.analysis_service
        history_service = current_app.history_service
        
        # Execute optimization
        result = analysis_service.execute_optimization(
            max_positions=max_positions,
            cash_reserve_pct=cash_reserve_pct,
            risk_profile=risk_profile
        )
        
        if result.get('success'):
            # Take a snapshot after optimization
            history_service.add_portfolio_snapshot()
            
            # Flash results
            flash(result.get('message', "Optimization executed successfully"), 'success')
            
            if result.get('errors'):
                for error in result.get('errors'):
                    flash(f"Error: {error.get('ticker', '')} - {error.get('error', '')}", 'warning')
        else:
            flash(result.get('message', 'Error executing optimization.'), 'danger')
        
        return redirect(url_for('web.index'))
        
    except Exception as e:
        logger.error(f"Error executing optimization: {e}", exc_info=True)
        flash(f'Error executing optimization: {str(e)}', 'danger')
        return redirect(url_for('web.optimize'))

@web_blueprint.route('/notifications', methods=['GET', 'POST'])
def notifications():
    """Notification settings page."""
    form_service = current_app.form_service
    notification_service = current_app.notification_service
    form = form_service.get_notification_form()
    
    # Load notification config
    config = notification_service.get_notification_config()
    
    if request.method == 'POST' and form.validate_on_submit():
        # Update notification config
        result = notification_service.update_notification_config(
            enabled=form.enabled.data,
            email_enabled=form.email_enabled.data,
            email_address=form.email_address.data,
            webhook_enabled=form.webhook_enabled.data,
            webhook_url=form.webhook_url.data,
            notify_trades=form.notify_trades.data,
            notify_thresholds=form.notify_thresholds.data
        )
        
        if result.get('success'):
            flash('Notification settings updated.', 'success')
        else:
            flash(result.get('message', 'Error saving notification settings.'), 'danger')
        
        return redirect(url_for('web.notifications'))
    
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

@web_blueprint.route('/indicators', methods=['GET', 'POST'])
def indicators():
    """Technical indicators analysis page."""
    form_service = current_app.form_service
    analysis_service = current_app.analysis_service
    form = form_service.get_indicator_analysis_form()
    analysis_results = None
    charts = {}
    
    if request.method == 'POST' and form.validate_on_submit():
        ticker = form.ticker.data.upper()
        days = form.days.data
        interval = form.interval.data
        
        try:
            # Analyze indicator data
            result = analysis_service.analyze_indicators(ticker, days, interval)
            
            if result.get('success'):
                analysis_results = result.get('results')
                charts = result.get('charts', {})
            else:
                flash(result.get('message', f'Error analyzing indicators for {ticker}.'), 'warning')
        
        except Exception as e:
            logger.error(f"Error analyzing indicators: {e}", exc_info=True)
            flash(f'Error in indicator analysis: {str(e)}', 'danger')
    
    return render_template('indicators.html', form=form, results=analysis_results, charts=charts)

@web_blueprint.route('/fundamentals', methods=['GET', 'POST'])
def fundamentals():
    """Fundamental analysis page."""
    form_service = current_app.form_service
    analysis_service = current_app.analysis_service
    form = form_service.get_fundamental_analysis_form()
    results = None
    
    if request.method == 'POST' and form.validate_on_submit():
        ticker = form.ticker.data.upper()
        
        try:
            # Get fundamental analysis
            result = analysis_service.analyze_fundamentals(ticker)
            
            if result.get('success'):
                results = result.get('results')
            else:
                flash(result.get('message', f'Unable to get fundamental data for {ticker}.'), 'warning')
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}", exc_info=True)
            flash(f'Error in fundamental analysis: {str(e)}', 'danger')
    
    return render_template('fundamentals.html', form=form, results=results)

@web_blueprint.route('/news', methods=['GET', 'POST'])
def news():
    """News and sentiment analysis page."""
    form_service = current_app.form_service
    analysis_service = current_app.analysis_service
    form = form_service.get_news_analysis_form()
    results = None
    
    if request.method == 'POST' and form.validate_on_submit():
        ticker = form.ticker.data.upper()
        
        try:
            # Get news sentiment analysis
            result = analysis_service.analyze_news(ticker)
            
            if result.get('success'):
                results = result.get('results')
            else:
                flash(result.get('message', f'No news found for {ticker}.'), 'warning')
            
        except Exception as e:
            logger.error(f"Error in news analysis: {e}", exc_info=True)
            flash(f'Error in news analysis: {str(e)}', 'danger')
    
    return render_template('news.html', form=form, results=results)

@web_blueprint.route('/market_sentiment')
def market_sentiment():
    """Market sentiment analysis page."""
    try:
        market_service = current_app.market_service
        
        # Force refresh if requested
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        sentiment = market_service.get_market_sentiment(force_refresh=force_refresh)
        
        return render_template('market_sentiment.html', sentiment=sentiment)
    except Exception as e:
        logger.error(f"Error getting market sentiment: {e}", exc_info=True)
        flash(f'Error getting market sentiment: {str(e)}', 'danger')
        return redirect(url_for('web.index'))

@web_blueprint.route('/parameters', methods=['GET', 'POST'])
def parameters():
    """Trading parameters settings page."""
    form_service = current_app.form_service
    trading_service = current_app.trading_service
    form = form_service.get_trading_params_form()
    
    if request.method == 'POST' and form.validate_on_submit():
        # Update trading parameters
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
            'trailing_stop_pct': form.trailing_stop_pct.data,
        }
        
        result = trading_service.update_trading_parameters(params)
        
        if result.get('success'):
            flash('Trading parameters updated successfully!', 'success')
        else:
            flash(result.get('message', 'Error saving parameters.'), 'danger')
        
        return redirect(url_for('web.parameters'))
    
    # Fill form with current values
    if request.method == 'GET':
        current_params = trading_service.get_trading_parameters()
        form.sma_period.data = current_params.get('sma_period', 20)
        form.ema_period.data = current_params.get('ema_period', 9)
        form.rsi_period.data = current_params.get('rsi_period', 14)
        form.macd_fast.data = current_params.get('macd_fast', 12)
        form.macd_slow.data = current_params.get('macd_slow', 26)
        form.macd_signal.data = current_params.get('macd_signal', 9)
        form.bb_window.data = current_params.get('bb_window', 20)
        form.bb_std.data = current_params.get('bb_std', 2)
        form.decision_buy_threshold.data = current_params.get('decision_buy_threshold', 60)
        form.decision_sell_threshold.data = current_params.get('decision_sell_threshold', -60)
        form.take_profit_pct.data = current_params.get('take_profit_pct', 5.0)
        form.stop_loss_pct.data = current_params.get('stop_loss_pct', -8.0)
        form.trailing_stop_pct.data = current_params.get('trailing_stop_pct', 3.0)
    
    return render_template('parameters.html', form=form, params=trading_service.get_trading_parameters())

@web_blueprint.route('/diagnostics')
def diagnostics():
    """System diagnostics page."""
    try:
        system_service = current_app.system_service
        
        # Run system diagnostics
        results = system_service.run_diagnostics()
        
        # Check for updates
        update_info = system_service.check_for_updates()
        
        return render_template('diagnostics.html', results=results, update_info=update_info, version=system_service.VERSION)
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}", exc_info=True)
        flash(f'Error running diagnostics: {str(e)}', 'danger')
        return redirect(url_for('web.index'))

@web_blueprint.route('/download_portfolio')
def download_portfolio():
    """Download portfolio as JSON file."""
    try:
        portfolio_service = current_app.portfolio_service
        data = portfolio_service.load_portfolio()
        
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
        logger.error(f"Error downloading portfolio: {e}", exc_info=True)
        flash(f'Error downloading portfolio: {str(e)}', 'danger')
        return redirect(url_for('web.index'))

@web_blueprint.route('/upload-json', methods=['GET', 'POST'])
def upload_json():
    """Upload portfolio from JSON file."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if file:
            try:
                portfolio_service = current_app.portfolio_service
                result = portfolio_service.import_portfolio_from_file(file)
                
                if result.get('success'):
                    flash('Portfolio loaded successfully!', 'success')
                    return redirect(url_for('web.index'))
                else:
                    flash(result.get('message', 'Error loading file.'), 'danger')
            except Exception as e:
                flash(f'Error loading file: {str(e)}', 'danger')
    
    return render_template('upload.html')

@web_blueprint.route('/clear_cache')
def clear_cache():
    """Admin route to clear all caches."""
    try:
        cache_service = current_app.cache_service
        cache_service.clear_cache()
        flash('Cache cleared successfully', 'success')
    except Exception as e:
        flash(f'Error clearing cache: {str(e)}', 'danger')
    
    return redirect(url_for('web.diagnostics'))

# API routes
@web_blueprint.route('/api/portfolio')
def api_portfolio():
    """API endpoint to get portfolio data."""
    portfolio_service = current_app.portfolio_service
    data = portfolio_service.load_portfolio()
    return jsonify(data)

@web_blueprint.route('/api/market_sentiment')
def api_market_sentiment():
    """API endpoint to get market sentiment."""
    try:
        market_service = current_app.market_service
        sentiment = market_service.get_market_sentiment()
        return jsonify(sentiment)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@web_blueprint.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint to analyze portfolio."""
    try:
        data = request.json
        portfolio = data.get('portfolio', {})
        account_balance = data.get('account_balance', 0)
        risk_profile = data.get('risk_profile', 'medium')
        quick_mode = data.get('quick_mode', True)
        
        analysis_service = current_app.analysis_service
        portfolio_analysis = analysis_service.analyze_portfolio(
            portfolio, account_balance, risk_profile, False, {}, quick_mode
        )
        
        return jsonify(portfolio_analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@web_blueprint.route('/api/ticker_data/<ticker>')
def api_ticker_data(ticker):
    """API endpoint to get ticker historical data."""
    try:
        days = int(request.args.get('days', 60))
        interval = request.args.get('interval', '1d')
        
        data_service = current_app.data_service
        result = data_service.get_ticker_data(ticker, days, interval)
        
        if result.get('success'):
            return jsonify(result.get('data'))
        else:
            return jsonify({"error": result.get('message', "No data available")}), 404
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@web_blueprint.route('/api/cache/status')
def api_cache_status():
    """API endpoint to check cache status."""
    try:
        cache_service = current_app.cache_service
        return jsonify(cache_service.get_cache_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500