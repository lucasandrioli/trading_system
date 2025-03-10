import logging
from flask import Flask
from flask_wtf.csrf import CSRFProtect

logger = logging.getLogger("trading_system.web")

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Default configuration
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATA_FOLDER='data',
        CACHE_EXPIRY=3600,  # 1 hour
        POLYGON_API_KEY='',
        DEBUG=False,
        TESTING=False,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max upload
    )
    
    # Override with any provided config
    if config:
        app.config.from_mapping(config)
    
    # Initialize CSRF protection
    csrf = CSRFProtect(app)
    
    # Register blueprints
    from trading_system.web.routes import web_blueprint
    app.register_blueprint(web_blueprint)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Initialize services
    with app.app_context():
        from trading_system.services import (
            PortfolioService, 
            DataService, 
            AnalysisService,
            MarketService,
            FormService,
            NotificationService,
            HistoryService,
            SystemService,
            CacheService,
            TradingService
        )
        
        # Create service instances
        app.cache_service = CacheService(app.config)
        app.data_service = DataService(app.config, app.cache_service)
        app.market_service = MarketService(app.config, app.data_service, app.cache_service)
        app.portfolio_service = PortfolioService(app.config, app.data_service)
        app.history_service = HistoryService(app.config, app.portfolio_service)
        app.notification_service = NotificationService(app.config)
        app.trading_service = TradingService(app.config, app.cache_service)
        app.form_service = FormService()
        app.system_service = SystemService(app.config)
        app.analysis_service = AnalysisService(
            app.config, 
            app.data_service, 
            app.portfolio_service, 
            app.market_service,
            app.trading_service,
            app.cache_service
        )
        
        # Log the app startup
        logger.info(f"Flask application initialized with {len(app.blueprints)} blueprints")
    
    return app

def register_error_handlers(app):
    """Register error handlers for the application."""
    
    @app.errorhandler(404)
    def handle_not_found(e):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def handle_server_error(e):
        logger.error(f"Server error: {e}", exc_info=True)
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(403)
    def handle_forbidden(e):
        return render_template('errors/403.html'), 403
    
    from flask import render_template
    
    logger.debug("Error handlers registered")