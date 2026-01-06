"""
Strategies Router

Endpoints for managing trading strategies.
"""

import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from hummingbot.api.dependencies import (
    get_hummingbot_application,
    require_no_strategy_running,
    require_strategy_running,
)
from hummingbot.api.models.base import APIResponse
from hummingbot.api.models.strategies import StrategyInfo, StrategyStartRequest, StrategyStatus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("", response_model=APIResponse[List[StrategyInfo]])
async def list_strategies():
    """
    List all available strategies and scripts.
    """
    try:
        strategies = []

        # List script strategies
        from hummingbot.client.config.config_helpers import SCRIPTS_PATH
        if os.path.exists(SCRIPTS_PATH):
            for filename in os.listdir(SCRIPTS_PATH):
                if filename.endswith('.py') and not filename.startswith('_'):
                    strategies.append(StrategyInfo(
                        name=filename.replace('.py', ''),
                        display_name=filename.replace('.py', '').replace('_', ' ').title(),
                        strategy_type="script",
                        description=f"Script strategy: {filename}"
                    ))

        # List v1 strategies
        from hummingbot.client.settings import STRATEGIES
        for strategy_name in STRATEGIES:
            strategies.append(StrategyInfo(
                name=strategy_name,
                display_name=strategy_name.replace('_', ' ').title(),
                strategy_type="v1",
                description=f"Built-in strategy: {strategy_name}"
            ))

        return APIResponse(data=strategies)
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=APIResponse[StrategyStatus])
async def get_strategy_status(app=Depends(get_hummingbot_application)):
    """
    Get the current strategy status.
    """
    try:
        if app.trading_core is None or app.trading_core.strategy is None:
            return APIResponse(data=StrategyStatus(
                is_running=False,
                status_text="No strategy running"
            ))

        strategy = app.trading_core.strategy
        strategy_name = app.trading_core.strategy_name

        # Determine strategy type
        strategy_type = "v1"
        if hasattr(strategy, 'controllers'):
            strategy_type = "v2"
        elif hasattr(strategy, 'on_tick'):
            strategy_type = "script"

        # Calculate runtime
        runtime_seconds = 0
        if hasattr(app.trading_core, 'strategy_task') and app.trading_core.strategy_task:
            # Estimate from clock
            if hasattr(app, 'clock') and app.clock:
                runtime_seconds = app.clock.current_timestamp - app.clock.start_time

        # Get status text
        status_text = None
        if hasattr(strategy, 'format_status'):
            try:
                status_text = strategy.format_status()
            except Exception:
                pass

        return APIResponse(data=StrategyStatus(
            is_running=True,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            runtime_seconds=runtime_seconds,
            status_text=status_text
        ))
    except Exception as e:
        logger.error(f"Error getting strategy status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=APIResponse)
async def start_strategy(
    request: StrategyStartRequest,
    app=Depends(require_no_strategy_running)
):
    """
    Start a strategy.
    """
    try:
        # Validate request
        if not request.script_name and not request.strategy_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either script_name or strategy_name is required"
            )

        if request.script_name:
            # Start script strategy
            from hummingbot.client.config.config_helpers import SCRIPTS_PATH

            script_path = os.path.join(SCRIPTS_PATH, f"{request.script_name}.py")
            if not os.path.exists(script_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Script '{request.script_name}' not found"
                )

            # Use the start command logic
            # This is a simplified version - full implementation would mirror start_command.py
            logger.info(f"Starting script: {request.script_name}")

            # Queue the start command
            if hasattr(app, '_notify_listeners'):
                app._notify_listeners('start', {
                    'script': request.script_name,
                    'config': request.config
                })

            return APIResponse(
                success=True,
                message=f"Script '{request.script_name}' start initiated"
            )
        else:
            # Start v1 strategy
            logger.info(f"Starting strategy: {request.strategy_name}")

            return APIResponse(
                success=True,
                message=f"Strategy '{request.strategy_name}' start initiated"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=APIResponse)
async def stop_strategy(
    skip_order_cancellation: bool = False,
    app=Depends(require_strategy_running)
):
    """
    Stop the currently running strategy.
    """
    try:
        strategy_name = app.strategy_name if hasattr(app, 'strategy_name') else "strategy"

        # Stop the strategy
        if hasattr(app, 'stop'):
            await app.stop()
        elif hasattr(app.trading_core, 'stop_strategy'):
            await app.trading_core.stop_strategy()

        return APIResponse(
            success=True,
            message=f"Strategy '{strategy_name}' stopped"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs", response_model=APIResponse[List[str]])
async def list_strategy_configs():
    """
    List available strategy configuration files.
    """
    try:
        from hummingbot.client.config.config_helpers import STRATEGIES_CONF_DIR_PATH

        configs = []
        if os.path.exists(STRATEGIES_CONF_DIR_PATH):
            for filename in os.listdir(STRATEGIES_CONF_DIR_PATH):
                if filename.endswith('.yml') or filename.endswith('.yaml'):
                    configs.append(filename)

        return APIResponse(data=configs)
    except Exception as e:
        logger.error(f"Error listing strategy configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs/{config_name}", response_model=APIResponse[Dict[str, Any]])
async def get_strategy_config(config_name: str):
    """
    Get a specific strategy configuration.
    """
    try:
        import yaml

        from hummingbot.client.config.config_helpers import STRATEGIES_CONF_DIR_PATH

        filepath = os.path.join(STRATEGIES_CONF_DIR_PATH, config_name)
        if not filepath.endswith('.yml') and not filepath.endswith('.yaml'):
            filepath += '.yml'

        if not os.path.exists(filepath):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Config '{config_name}' not found"
            )

        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        return APIResponse(data=config_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
