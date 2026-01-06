"""
FastAPI Dependencies

Provides dependency injection for accessing Hummingbot internals.
"""

import logging

from fastapi import Depends, HTTPException, status

logger = logging.getLogger(__name__)


def get_hummingbot_application():
    """
    Dependency to get the HummingbotApplication singleton.
    Returns None if not initialized yet.
    """
    try:
        from hummingbot.client.hummingbot_application import HummingbotApplication
        app = HummingbotApplication.main_application()
        if app is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Hummingbot application not initialized"
            )
        return app
    except Exception as e:
        logger.error(f"Error getting HummingbotApplication: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Hummingbot application not available: {str(e)}"
        )


def get_trading_core(app=Depends(get_hummingbot_application)):
    """Dependency to get the TradingCore instance"""
    if app.trading_core is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trading core not initialized"
        )
    return app.trading_core


def get_security():
    """Dependency to get the Security singleton for credential management"""
    from hummingbot.client.config.security import Security
    return Security


def get_client_config():
    """Dependency to get the client configuration"""
    from hummingbot.client.config.client_config_map import ClientConfigMap
    from hummingbot.client.config.config_helpers import ClientConfigAdapter
    try:
        from hummingbot.client.hummingbot_application import HummingbotApplication
        app = HummingbotApplication.main_application()
        if app and hasattr(app, 'client_config_map'):
            return app.client_config_map
    except Exception:
        pass
    # Return default config if app not available
    return ClientConfigAdapter(ClientConfigMap())


def require_strategy_running(app=Depends(get_hummingbot_application)):
    """Dependency that requires a strategy to be running"""
    if app.trading_core is None or app.trading_core.strategy is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No strategy is currently running"
        )
    return app.trading_core


def require_no_strategy_running(app=Depends(get_hummingbot_application)):
    """Dependency that requires no strategy to be running"""
    if app.trading_core is not None and app.trading_core.strategy is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A strategy is already running. Stop it first."
        )
    return app
