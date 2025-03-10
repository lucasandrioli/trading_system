import logging
from flask import Blueprint, jsonify, request, current_app
from trading_system.core.analysis.market_analysis import MarketAnalysis
from trading_system.core.data.data_loader import DataLoader

# Configure logging
logger = logging.getLogger("trading_system.api.routes")

# Create API blueprint
api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/portfolio')
def api_portfolio():
    """API endpoint to get portfolio data."""
    portfolio_service = current_app.portfolio_service
    data = portfolio_service.load_portfolio()
    return jsonify(data)

@api_blueprint.route('/market_sentiment')
def api_market_sentiment():
    """API endpoint to get market sentiment data."""
    try:
        # Use cache manager if available
        cache_manager = getattr(current_app, 'cache_manager', None)
        
        if cache_manager:
            sentiment = cache_manager.get('market_sentiment', expiry=1800)  # 30-minute cache
            if sentiment:
                return jsonify(sentiment)
        
        # Fresh analysis if not cached
        sentiment = MarketAnalysis.get_market_sentiment()
        
        if cache_manager:
            cache_manager.set('market_sentiment', sentiment)
        
        return jsonify(sentiment)
    
    except Exception as e:
        logger.error(f"Error in market sentiment API: {e}")
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/ticker_data/<ticker>')
def api_ticker_data(ticker):
    """API endpoint to get historical data for a ticker."""
    try:
        days = int(request.args.get('days', 60))
        interval = request.args.get('interval', '1d')
        
        # Check cache if available
        cache_manager = getattr(current_app, 'cache_manager', None)
        cache_key = f"ticker_data:{ticker}:{days}:{interval}"
        
        if cache_manager:
            cached_data = cache_manager.get(cache_key, expiry=3600)  # 1-hour cache
            if cached_data:
                return jsonify(cached_data)
        
        # Fetch fresh data
        df = DataLoader.get_asset_data(ticker, days, interval)
        
        if df.empty:
            return jsonify({"error": "No data available for this ticker"}), 404
        
        # Convert to JSON-friendly format
        df_json = df.reset_index()
        if 'Date' in df_json.columns:
            df_json['Date'] = df_json['Date'].dt.strftime('%Y-%m-%d')
        
        data = df_json.to_dict(orient='records')
        
        # Cache the result
        if cache_manager:
            cache_manager.set(cache_key, data)
        
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error in ticker data API for {ticker}: {e}")
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/analyze', methods=['POST'])
def api_analyze():
    """API endpoint to analyze a portfolio."""
    from trading_system.core.strategy.strategy import Strategy
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        portfolio = data.get('portfolio', {})
        account_balance = data.get('account_balance', 0)
        risk_profile = data.get('risk_profile', 'medium')
        
        # Check cache if available
        cache_manager = getattr(current_app, 'cache_manager', None)
        cache_key = f"analysis:{hash(str(portfolio))}-{account_balance}-{risk_profile}"
        
        if cache_manager:
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return jsonify(cached_result)
        
        # Perform analysis
        from trading_system.services.analysis_service import analyze_portfolio
        
        analysis_result = analyze_portfolio(
            portfolio, account_balance, risk_profile, {}, False
        )
        
        # Cache result
        if cache_manager:
            cache_manager.set(cache_key, analysis_result)
        
        return jsonify(analysis_result)
    
    except Exception as e:
        logger.error(f"Error in portfolio analysis API: {e}")
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/cache/status')
def api_cache_status():
    """API endpoint to check cache status."""
    try:
        cache_manager = getattr(current_app, 'cache_manager', None)
        
        if not cache_manager:
            return jsonify({"status": "Cache not available"}), 404
        
        import sys
        
        status = {
            "cache_size": len(cache_manager.cache),
            "memory_usage_mb": sys.getsizeof(cache_manager.cache) / (1024 * 1024),
            "items_count": len(cache_manager.cache),
            "oldest_item_age": min(cache_manager.timestamps.values()) if cache_manager.timestamps else None,
            "newest_item_age": max(cache_manager.timestamps.values()) if cache_manager.timestamps else None
        }
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error in cache status API: {e}")
        return jsonify({"error": str(e)}), 500