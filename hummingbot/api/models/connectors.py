"""
Connector-related models
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConnectorCredentials(BaseModel):
    """Model for connector credentials"""
    connector_name: str = Field(..., description="Name of the connector (e.g., 'binance', 'kucoin')")
    credentials: Dict[str, Any] = Field(..., description="Credential key-value pairs")

    model_config = {"json_schema_extra": {
        "example": {
            "connector_name": "binance",
            "credentials": {
                "binance_api_key": "your-api-key",
                "binance_api_secret": "your-api-secret"
            }
        }
    }}


class ConnectorInfo(BaseModel):
    """Model for connector information"""
    name: str
    display_name: str
    connector_type: str  # "exchange", "dex", "gateway"
    is_configured: bool = False
    is_connected: bool = False
    trading_pairs: List[str] = []


class ConnectorBalance(BaseModel):
    """Model for connector balance"""
    asset: str
    total: Decimal
    available: Decimal
    locked: Decimal = Decimal("0")


class ConnectorStatus(BaseModel):
    """Model for connector status"""
    name: str
    is_ready: bool
    status_dict: Dict[str, bool] = {}
    balances: List[ConnectorBalance] = []
    trading_pairs: List[str] = []
    error: Optional[str] = None


class AvailableConnector(BaseModel):
    """Model for available connector info"""
    name: str
    display_name: str
    connector_type: str
    required_credentials: List[str] = []
    is_configured: bool = False
