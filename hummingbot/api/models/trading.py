"""
Trading-related models
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field


class OrderInfo(BaseModel):
    """Model for order information"""
    order_id: str
    client_order_id: str
    connector_name: str
    trading_pair: str
    order_type: str  # "limit", "market"
    side: str  # "buy", "sell"
    price: Optional[Decimal] = None
    amount: Decimal
    filled_amount: Decimal = Decimal("0")
    status: str  # "open", "filled", "cancelled", "failed"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PlaceOrderRequest(BaseModel):
    """Request model for placing an order"""
    connector_name: str
    trading_pair: str
    side: str = Field(..., description="'buy' or 'sell'")
    order_type: str = Field("limit", description="'limit' or 'market'")
    amount: Decimal
    price: Optional[Decimal] = Field(None, description="Required for limit orders")

    model_config = {"json_schema_extra": {
        "example": {
            "connector_name": "binance",
            "trading_pair": "BTC-USDT",
            "side": "buy",
            "order_type": "limit",
            "amount": 0.001,
            "price": 50000
        }
    }}


class CancelOrderRequest(BaseModel):
    """Request model for cancelling an order"""
    connector_name: str
    trading_pair: str
    order_id: str


class PositionInfo(BaseModel):
    """Model for position information (perpetuals)"""
    connector_name: str
    trading_pair: str
    position_side: str  # "long", "short"
    amount: Decimal
    entry_price: Decimal
    mark_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    leverage: Optional[int] = None
    liquidation_price: Optional[Decimal] = None


class TradeInfo(BaseModel):
    """Model for trade information"""
    trade_id: str
    order_id: str
    connector_name: str
    trading_pair: str
    side: str
    price: Decimal
    amount: Decimal
    fee: Optional[Decimal] = None
    fee_asset: Optional[str] = None
    timestamp: datetime


class MarketPrice(BaseModel):
    """Model for market price"""
    connector_name: str
    trading_pair: str
    mid_price: Decimal
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    last_price: Optional[Decimal] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
