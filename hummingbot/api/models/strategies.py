"""
Strategy-related models
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StrategyStartRequest(BaseModel):
    """Request model for starting a strategy"""
    strategy_name: Optional[str] = Field(None, description="Strategy name (for v1 strategies)")
    script_name: Optional[str] = Field(None, description="Script file name (for script strategies)")
    config_file: Optional[str] = Field(None, description="Config file name")
    config: Optional[Dict[str, Any]] = Field(None, description="Inline configuration")

    model_config = {"json_schema_extra": {
        "example": {
            "script_name": "simple_pmm",
            "config": {
                "exchange": "binance_paper_trade",
                "trading_pair": "BTC-USDT",
                "order_amount": 0.001
            }
        }
    }}


class StrategyStatus(BaseModel):
    """Model for strategy status"""
    is_running: bool = False
    strategy_name: Optional[str] = None
    strategy_type: Optional[str] = None  # "v1", "v2", "script"
    config_file: Optional[str] = None
    start_time: Optional[datetime] = None
    runtime_seconds: float = 0
    status_text: Optional[str] = None


class StrategyInfo(BaseModel):
    """Model for available strategy info"""
    name: str
    display_name: str
    strategy_type: str  # "v1", "v2", "script"
    description: Optional[str] = None
    required_connectors: List[str] = []


class StrategyPerformance(BaseModel):
    """Model for strategy performance"""
    total_pnl: Decimal = Decimal("0")
    total_pnl_pct: Decimal = Decimal("0")
    volume_traded: Decimal = Decimal("0")
    trades_count: int = 0
    win_rate: Optional[Decimal] = None
    start_time: Optional[datetime] = None
    runtime_seconds: float = 0
