"""
Connectors Router

Endpoints for managing exchange connectors and credentials.
"""

import logging
from decimal import Decimal
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from hummingbot.api.dependencies import get_hummingbot_application, get_security
from hummingbot.api.models.base import APIResponse
from hummingbot.api.models.connectors import AvailableConnector, ConnectorBalance, ConnectorCredentials, ConnectorStatus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("", response_model=APIResponse[List[AvailableConnector]])
async def list_connectors():
    """
    List all available connectors and their configuration status.
    """
    try:
        from hummingbot.client.config.security import Security
        from hummingbot.client.settings import AllConnectorSettings

        all_connectors = AllConnectorSettings.get_connector_settings()
        configured_connectors = set()

        # Get list of configured connectors
        try:
            if Security.is_decryption_done():
                configured_connectors = set(Security.all_decrypted_values().keys())
        except Exception:
            pass

        result = []
        for name, settings in all_connectors.items():
            connector_info = AvailableConnector(
                name=name,
                display_name=settings.display_name if hasattr(settings, 'display_name') else name,
                connector_type=settings.type.name if hasattr(settings, 'type') else "exchange",
                required_credentials=list(settings.config_keys.keys()) if hasattr(settings, 'config_keys') else [],
                is_configured=name in configured_connectors
            )
            result.append(connector_info)

        return APIResponse(data=result)
    except Exception as e:
        logger.error(f"Error listing connectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{connector_name}/status", response_model=APIResponse[ConnectorStatus])
async def get_connector_status(
    connector_name: str,
    app=Depends(get_hummingbot_application)
):
    """
    Get the status of a specific connector.
    """
    try:
        # Check if connector exists in markets
        if connector_name not in app.markets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Connector '{connector_name}' is not active"
            )

        connector = app.markets[connector_name]

        # Get balances
        balances = []
        for asset, balance in connector.get_all_balances().items():
            available = connector.get_available_balance(asset)
            balances.append(ConnectorBalance(
                asset=asset,
                total=Decimal(str(balance)),
                available=Decimal(str(available)),
                locked=Decimal(str(balance - available))
            ))

        # Get trading pairs
        trading_pairs = list(connector.trading_pairs) if hasattr(connector, 'trading_pairs') else []

        # Get status dict
        status_dict = connector.status_dict if hasattr(connector, 'status_dict') else {}

        return APIResponse(data=ConnectorStatus(
            name=connector_name,
            is_ready=connector.ready,
            status_dict=status_dict,
            balances=balances,
            trading_pairs=trading_pairs
        ))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting connector status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{connector_name}/balances", response_model=APIResponse[List[ConnectorBalance]])
async def get_connector_balances(
    connector_name: str,
    app=Depends(get_hummingbot_application)
):
    """
    Get balances for a specific connector.
    """
    try:
        if connector_name not in app.markets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Connector '{connector_name}' is not active"
            )

        connector = app.markets[connector_name]
        balances = []

        for asset, balance in connector.get_all_balances().items():
            available = connector.get_available_balance(asset)
            balances.append(ConnectorBalance(
                asset=asset,
                total=Decimal(str(balance)),
                available=Decimal(str(available)),
                locked=Decimal(str(balance - available))
            ))

        return APIResponse(data=balances)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting balances: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{connector_name}/credentials", response_model=APIResponse)
async def set_connector_credentials(
    connector_name: str,
    credentials: ConnectorCredentials,
    security=Depends(get_security)
):
    """
    Set credentials for a connector.
    """
    try:
        from hummingbot.client.config.config_helpers import get_connector_config_map
        from hummingbot.client.settings import AllConnectorSettings

        # Verify connector exists
        all_connectors = AllConnectorSettings.get_connector_settings()
        if connector_name not in all_connectors:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown connector: {connector_name}"
            )

        # Get the config map for this connector
        config_map = get_connector_config_map(connector_name)
        if config_map is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not get config map for connector: {connector_name}"
            )

        # Update credentials
        for key, value in credentials.credentials.items():
            if hasattr(config_map, key):
                setattr(config_map, key, value)
            else:
                logger.warning(f"Unknown credential key: {key}")

        # Save encrypted config
        security.update_secure_config(config_map)

        return APIResponse(
            success=True,
            message=f"Credentials saved for {connector_name}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting credentials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{connector_name}/credentials", response_model=APIResponse)
async def delete_connector_credentials(
    connector_name: str,
    security=Depends(get_security)
):
    """
    Delete credentials for a connector.
    """
    try:
        import os

        from hummingbot.client.config.config_helpers import CONNECTORS_CONF_DIR_PATH

        config_file = os.path.join(CONNECTORS_CONF_DIR_PATH, f"{connector_name}.yml")

        if os.path.exists(config_file):
            os.remove(config_file)
            return APIResponse(
                success=True,
                message=f"Credentials deleted for {connector_name}"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No credentials found for {connector_name}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting credentials: {e}")
        raise HTTPException(status_code=500, detail=str(e))
