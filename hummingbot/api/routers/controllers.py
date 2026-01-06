"""
Controllers Router

Endpoints for managing trading controllers (v2 strategies).
"""

import logging
import os
from decimal import Decimal
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from hummingbot.api.dependencies import get_hummingbot_application
from hummingbot.api.models.base import APIResponse
from hummingbot.api.models.controllers import (
    ControllerConfig,
    ControllerCreateRequest,
    ControllerStatus,
    ControllerUpdateRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_controllers_config_dir() -> str:
    """Get the controllers configuration directory"""
    from hummingbot.client.config.config_helpers import CONTROLLERS_CONF_DIR_PATH
    return CONTROLLERS_CONF_DIR_PATH


@router.get("", response_model=APIResponse[List[ControllerConfig]])
async def list_controllers():
    """
    List all available controller configurations.
    """
    try:
        import yaml
        config_dir = get_controllers_config_dir()

        if not os.path.exists(config_dir):
            return APIResponse(data=[])

        controllers = []
        for filename in os.listdir(config_dir):
            if filename.endswith('.yml') or filename.endswith('.yaml'):
                filepath = os.path.join(config_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        config_data = yaml.safe_load(f)
                        if config_data:
                            controller = ControllerConfig(
                                id=filename.replace('.yml', '').replace('.yaml', ''),
                                controller_name=config_data.get('controller_name', ''),
                                controller_type=config_data.get('controller_type', 'generic'),
                                connector_name=config_data.get('connector_name'),
                                trading_pair=config_data.get('trading_pair'),
                                total_amount_quote=Decimal(str(config_data.get('total_amount_quote', 0))) if config_data.get('total_amount_quote') else None,
                                manual_kill_switch=config_data.get('manual_kill_switch', False),
                                config=config_data
                            )
                            controllers.append(controller)
                except Exception as e:
                    logger.warning(f"Error reading controller config {filename}: {e}")

        return APIResponse(data=controllers)
    except Exception as e:
        logger.error(f"Error listing controllers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types", response_model=APIResponse[List[Dict[str, Any]]])
async def list_controller_types():
    """
    List all available controller types and their parameters.
    """
    try:
        # Get available controller types from the controllers directory
        controller_types = []

        # Market making controllers
        controller_types.append({
            "type": "market_making",
            "description": "Market making strategies that provide liquidity",
            "examples": ["pmm_simple", "pmm_dynamic", "cross_exchange_market_making"]
        })

        # Directional trading controllers
        controller_types.append({
            "type": "directional_trading",
            "description": "Directional strategies based on signals",
            "examples": ["macd_bb", "bollinger", "dman"]
        })

        # Generic controllers
        controller_types.append({
            "type": "generic",
            "description": "Generic strategies like grid trading, arbitrage",
            "examples": ["grid_strike", "xemm", "arbitrage"]
        })

        return APIResponse(data=controller_types)
    except Exception as e:
        logger.error(f"Error listing controller types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{controller_id}", response_model=APIResponse[ControllerConfig])
async def get_controller(controller_id: str):
    """
    Get a specific controller configuration.
    """
    try:
        import yaml
        config_dir = get_controllers_config_dir()
        filepath = os.path.join(config_dir, f"{controller_id}.yml")

        if not os.path.exists(filepath):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Controller '{controller_id}' not found"
            )

        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        controller = ControllerConfig(
            id=controller_id,
            controller_name=config_data.get('controller_name', ''),
            controller_type=config_data.get('controller_type', 'generic'),
            connector_name=config_data.get('connector_name'),
            trading_pair=config_data.get('trading_pair'),
            total_amount_quote=Decimal(str(config_data.get('total_amount_quote', 0))) if config_data.get('total_amount_quote') else None,
            manual_kill_switch=config_data.get('manual_kill_switch', False),
            config=config_data
        )

        return APIResponse(data=controller)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting controller: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=APIResponse[ControllerConfig])
async def create_controller(request: ControllerCreateRequest):
    """
    Create a new controller configuration.
    """
    try:
        import yaml
        config_dir = get_controllers_config_dir()
        os.makedirs(config_dir, exist_ok=True)

        # Generate ID from controller name and timestamp
        import time
        controller_id = f"{request.controller_name}_{int(time.time())}"
        filepath = os.path.join(config_dir, f"{controller_id}.yml")

        # Build config
        config_data = {
            'controller_name': request.controller_name,
            'controller_type': request.controller_type,
            **request.config
        }

        # Save config
        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

        controller = ControllerConfig(
            id=controller_id,
            controller_name=request.controller_name,
            controller_type=request.controller_type,
            connector_name=config_data.get('connector_name'),
            trading_pair=config_data.get('trading_pair'),
            total_amount_quote=Decimal(str(config_data.get('total_amount_quote', 0))) if config_data.get('total_amount_quote') else None,
            config=config_data
        )

        return APIResponse(
            data=controller,
            message=f"Controller '{controller_id}' created successfully"
        )
    except Exception as e:
        logger.error(f"Error creating controller: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{controller_id}", response_model=APIResponse[ControllerConfig])
async def update_controller(controller_id: str, request: ControllerUpdateRequest):
    """
    Update a controller configuration.
    """
    try:
        import yaml
        config_dir = get_controllers_config_dir()
        filepath = os.path.join(config_dir, f"{controller_id}.yml")

        if not os.path.exists(filepath):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Controller '{controller_id}' not found"
            )

        # Load existing config
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        # Update with new values
        config_data.update(request.config)

        # Save updated config
        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

        controller = ControllerConfig(
            id=controller_id,
            controller_name=config_data.get('controller_name', ''),
            controller_type=config_data.get('controller_type', 'generic'),
            connector_name=config_data.get('connector_name'),
            trading_pair=config_data.get('trading_pair'),
            total_amount_quote=Decimal(str(config_data.get('total_amount_quote', 0))) if config_data.get('total_amount_quote') else None,
            manual_kill_switch=config_data.get('manual_kill_switch', False),
            config=config_data
        )

        return APIResponse(
            data=controller,
            message=f"Controller '{controller_id}' updated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating controller: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{controller_id}", response_model=APIResponse)
async def delete_controller(controller_id: str):
    """
    Delete a controller configuration.
    """
    try:
        config_dir = get_controllers_config_dir()
        filepath = os.path.join(config_dir, f"{controller_id}.yml")

        if not os.path.exists(filepath):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Controller '{controller_id}' not found"
            )

        os.remove(filepath)

        return APIResponse(
            success=True,
            message=f"Controller '{controller_id}' deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting controller: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{controller_id}/status", response_model=APIResponse[ControllerStatus])
async def get_controller_status(
    controller_id: str,
    app=Depends(get_hummingbot_application)
):
    """
    Get the runtime status of a controller.
    """
    try:
        # Check if strategy is running and has this controller
        if app.trading_core is None or app.trading_core.strategy is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No strategy is currently running"
            )

        strategy = app.trading_core.strategy

        # Check if this is a v2 strategy with controllers
        if not hasattr(strategy, 'controllers'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Running strategy does not support controllers"
            )

        # Find the controller
        controller = None
        for c in strategy.controllers.values():
            if c.config.id == controller_id:
                controller = c
                break

        if controller is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Controller '{controller_id}' not found in running strategy"
            )

        # Get status
        status_info = ControllerStatus(
            id=controller_id,
            controller_name=controller.config.controller_name,
            controller_type=controller.config.controller_type,
            is_running=controller.is_running if hasattr(controller, 'is_running') else True,
            is_terminated=controller.is_terminated if hasattr(controller, 'is_terminated') else False,
            connector_name=controller.config.connector_name if hasattr(controller.config, 'connector_name') else None,
            trading_pair=controller.config.trading_pair if hasattr(controller.config, 'trading_pair') else None,
            status_text=controller.to_format_status() if hasattr(controller, 'to_format_status') else None
        )

        return APIResponse(data=status_info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting controller status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
