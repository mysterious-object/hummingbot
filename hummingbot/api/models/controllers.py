"""
Controller-related models
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ControllerCreateRequest(BaseModel):
    """Request model for creating a controller"""
    controller_name: str = Field(..., description="Name of the controller class")
    controller_type: str = Field(..., description="Type: 'market_making', 'directional_trading', 'generic'")
    config: Dict[str, Any] = Field(..., description="Controller configuration parameters")

    model_config = {"json_schema_extra": {
        "example": {
            "controller_name": "pmm_simple",
            "controller_type": "market_making",
            "config": {
                "connector_name": "binance",
                "trading_pair": "BTC-USDT",
                "total_amount_quote": 1000,
                "bid_spread": 0.001,
                "ask_spread": 0.001
            }
        }
    }}


class ControllerUpdateRequest(BaseModel):
    """Request model for updating a controller"""
    config: Dict[str, Any] = Field(..., description="Fields to update (only updatable fields)")


class ControllerConfig(BaseModel):
    """Model for controller configuration"""
    id: str
    controller_name: str
    controller_type: str
    connector_name: Optional[str] = None
    trading_pair: Optional[str] = None
    total_amount_quote: Optional[Decimal] = None
    manual_kill_switch: bool = False
    config: Dict[str, Any] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ControllerStatus(BaseModel):
    """Model for controller status"""
    id: str
    controller_name: str
    controller_type: str
    is_running: bool = False
    is_terminated: bool = False
    connector_name: Optional[str] = None
    trading_pair: Optional[str] = None
    pnl: Optional[Decimal] = None
    volume_traded: Optional[Decimal] = None
    open_orders: int = 0
    status_text: Optional[str] = None
    error: Optional[str] = None


class ControllerPerformance(BaseModel):
    """Model for controller performance metrics"""
    id: str
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    volume_traded: Decimal = Decimal("0")
    trades_count: int = 0
    open_orders_count: int = 0
    start_time: Optional[datetime] = None
    runtime_seconds: float = 0
