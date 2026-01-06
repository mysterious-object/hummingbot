"""
System Router

Endpoints for system status, configuration, and health checks.
"""

import logging
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from hummingbot.api.dependencies import get_client_config, get_hummingbot_application
from hummingbot.api.models.base import APIResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=APIResponse[Dict[str, Any]])
async def health_check():
    """
    Health check endpoint.
    """
    try:
        # Check if HummingbotApplication is available
        app_status = "unavailable"
        try:
            from hummingbot.client.hummingbot_application import HummingbotApplication
            app = HummingbotApplication.main_application()
            if app:
                app_status = "running"
        except Exception:
            pass

        return APIResponse(data={
            "status": "healthy",
            "application": app_status,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return APIResponse(
            success=False,
            data={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/info", response_model=APIResponse[Dict[str, Any]])
async def get_system_info():
    """
    Get system information.
    """
    try:
        # Get version
        version = "unknown"
        try:
            from hummingbot import VERSION
            version = VERSION
        except ImportError:
            pass

        return APIResponse(data={
            "version": version,
            "python_version": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=APIResponse[Dict[str, Any]])
async def get_status(app=Depends(get_hummingbot_application)):
    """
    Get overall bot status.
    """
    try:
        # Strategy status
        strategy_running = False
        strategy_name = None
        if app.trading_core and app.trading_core.strategy:
            strategy_running = True
            strategy_name = app.trading_core.strategy_name

        # Connector status
        connectors = {}
        for name, connector in app.markets.items():
            connectors[name] = {
                "ready": connector.ready,
                "trading_pairs": list(connector.trading_pairs) if hasattr(connector, 'trading_pairs') else []
            }

        return APIResponse(data={
            "strategy_running": strategy_running,
            "strategy_name": strategy_name,
            "connectors": connectors,
            "timestamp": datetime.utcnow().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=APIResponse[Dict[str, Any]])
async def get_config(
    keys: Optional[str] = None,
    config=Depends(get_client_config)
):
    """
    Get client configuration.

    Args:
        keys: Comma-separated list of config keys to retrieve (optional)
    """
    try:
        # Convert config to dict
        config_dict = {}

        if hasattr(config, 'dict'):
            config_dict = config.dict()
        elif hasattr(config, '__dict__'):
            config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}

        # Filter keys if specified
        if keys:
            key_list = [k.strip() for k in keys.split(',')]
            config_dict = {k: v for k, v in config_dict.items() if k in key_list}

        return APIResponse(data=config_dict)
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config", response_model=APIResponse)
async def update_config(
    updates: Dict[str, Any],
    config=Depends(get_client_config)
):
    """
    Update client configuration.
    """
    try:
        updated_keys = []

        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
                updated_keys.append(key)
            else:
                logger.warning(f"Unknown config key: {key}")

        # Save config
        from hummingbot.client.config.config_helpers import save_to_yml
        save_to_yml(config, config.config_file_name if hasattr(config, 'config_file_name') else 'conf_client.yml')

        return APIResponse(
            success=True,
            message=f"Updated config keys: {', '.join(updated_keys)}"
        )
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs", response_model=APIResponse[List[str]])
async def get_logs(
    lines: int = 100,
    level: Optional[str] = None
):
    """
    Get recent log entries.
    """
    try:
        from hummingbot.client.config.config_helpers import LOGS_PATH

        log_file = os.path.join(LOGS_PATH, "logs_hummingbot.log")

        if not os.path.exists(log_file):
            return APIResponse(data=[])

        # Read last N lines
        with open(log_file, 'r') as f:
            all_lines = f.readlines()

        # Filter by level if specified
        if level:
            level_upper = level.upper()
            all_lines = [line for line in all_lines if level_upper in line]

        # Get last N lines
        log_lines = all_lines[-lines:]

        return APIResponse(data=[line.strip() for line in log_lines])
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/markets", response_model=APIResponse[Dict[str, List[str]]])
async def list_markets(app=Depends(get_hummingbot_application)):
    """
    List all active markets/connectors and their trading pairs.
    """
    try:
        markets = {}
        for name, connector in app.markets.items():
            trading_pairs = list(connector.trading_pairs) if hasattr(connector, 'trading_pairs') else []
            markets[name] = trading_pairs

        return APIResponse(data=markets)
    except Exception as e:
        logger.error(f"Error listing markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))
