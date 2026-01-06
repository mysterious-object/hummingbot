"""
Trading Router

Endpoints for trading operations - orders, positions, market data.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from hummingbot.api.dependencies import get_hummingbot_application
from hummingbot.api.models.base import APIResponse
from hummingbot.api.models.trading import MarketPrice, OrderInfo, PlaceOrderRequest, PositionInfo

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/orders", response_model=APIResponse[List[OrderInfo]])
async def list_orders(
    connector_name: Optional[str] = None,
    trading_pair: Optional[str] = None,
    app=Depends(get_hummingbot_application)
):
    """
    List all open orders across connectors.
    """
    try:
        orders = []

        # Get orders from all active connectors
        for name, connector in app.markets.items():
            if connector_name and name != connector_name:
                continue

            # Get in-flight orders
            if hasattr(connector, 'in_flight_orders'):
                for order_id, order in connector.in_flight_orders.items():
                    if trading_pair and order.trading_pair != trading_pair:
                        continue

                    orders.append(OrderInfo(
                        order_id=order.exchange_order_id or "",
                        client_order_id=order.client_order_id,
                        connector_name=name,
                        trading_pair=order.trading_pair,
                        order_type=order.order_type.name.lower() if hasattr(order.order_type, 'name') else str(order.order_type),
                        side="buy" if order.is_buy else "sell",
                        price=Decimal(str(order.price)) if order.price else None,
                        amount=Decimal(str(order.amount)),
                        filled_amount=Decimal(str(order.executed_amount_base)) if hasattr(order, 'executed_amount_base') else Decimal("0"),
                        status=order.current_state.name.lower() if hasattr(order.current_state, 'name') else "open",
                        created_at=datetime.fromtimestamp(order.creation_timestamp) if hasattr(order, 'creation_timestamp') else None
                    ))

        return APIResponse(data=orders)
    except Exception as e:
        logger.error(f"Error listing orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orders", response_model=APIResponse[OrderInfo])
async def place_order(
    request: PlaceOrderRequest,
    app=Depends(get_hummingbot_application)
):
    """
    Place a new order.
    """
    try:
        if request.connector_name not in app.markets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Connector '{request.connector_name}' not active"
            )

        connector = app.markets[request.connector_name]

        # Validate order type and price
        if request.order_type == "limit" and request.price is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Price is required for limit orders"
            )

        from hummingbot.core.data_type.common import OrderType, TradeType

        order_type = OrderType.LIMIT if request.order_type == "limit" else OrderType.MARKET
        trade_type = TradeType.BUY if request.side.lower() == "buy" else TradeType.SELL

        # Place order
        if trade_type == TradeType.BUY:
            order_id = connector.buy(
                trading_pair=request.trading_pair,
                amount=request.amount,
                order_type=order_type,
                price=request.price
            )
        else:
            order_id = connector.sell(
                trading_pair=request.trading_pair,
                amount=request.amount,
                order_type=order_type,
                price=request.price
            )

        return APIResponse(
            data=OrderInfo(
                order_id="",
                client_order_id=order_id,
                connector_name=request.connector_name,
                trading_pair=request.trading_pair,
                order_type=request.order_type,
                side=request.side,
                price=request.price,
                amount=request.amount,
                status="pending"
            ),
            message=f"Order placed with ID: {order_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/orders/{order_id}", response_model=APIResponse)
async def cancel_order(
    order_id: str,
    connector_name: str,
    trading_pair: str,
    app=Depends(get_hummingbot_application)
):
    """
    Cancel an order.
    """
    try:
        if connector_name not in app.markets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Connector '{connector_name}' not active"
            )

        connector = app.markets[connector_name]
        connector.cancel(trading_pair=trading_pair, client_order_id=order_id)

        return APIResponse(
            success=True,
            message=f"Cancel request sent for order {order_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions", response_model=APIResponse[List[PositionInfo]])
async def list_positions(
    connector_name: Optional[str] = None,
    app=Depends(get_hummingbot_application)
):
    """
    List all open positions (for perpetual connectors).
    """
    try:
        positions = []

        for name, connector in app.markets.items():
            if connector_name and name != connector_name:
                continue

            # Check if this is a perpetual connector
            if hasattr(connector, 'account_positions'):
                for trading_pair, position in connector.account_positions.items():
                    positions.append(PositionInfo(
                        connector_name=name,
                        trading_pair=trading_pair,
                        position_side=position.position_side.name.lower() if hasattr(position.position_side, 'name') else "long",
                        amount=Decimal(str(position.amount)),
                        entry_price=Decimal(str(position.entry_price)),
                        unrealized_pnl=Decimal(str(position.unrealized_pnl)) if hasattr(position, 'unrealized_pnl') else None,
                        leverage=position.leverage if hasattr(position, 'leverage') else None
                    ))

        return APIResponse(data=positions)
    except Exception as e:
        logger.error(f"Error listing positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prices/{connector_name}/{trading_pair}", response_model=APIResponse[MarketPrice])
async def get_price(
    connector_name: str,
    trading_pair: str,
    app=Depends(get_hummingbot_application)
):
    """
    Get current market price for a trading pair.
    """
    try:
        if connector_name not in app.markets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Connector '{connector_name}' not active"
            )

        connector = app.markets[connector_name]

        # Get prices
        mid_price = connector.get_mid_price(trading_pair)
        bid_price = None
        ask_price = None

        if hasattr(connector, 'get_price'):
            bid_price = connector.get_price(trading_pair, is_buy=True)
            ask_price = connector.get_price(trading_pair, is_buy=False)

        return APIResponse(data=MarketPrice(
            connector_name=connector_name,
            trading_pair=trading_pair,
            mid_price=Decimal(str(mid_price)) if mid_price else Decimal("0"),
            bid_price=Decimal(str(bid_price)) if bid_price else None,
            ask_price=Decimal(str(ask_price)) if ask_price else None
        ))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting price: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orderbook/{connector_name}/{trading_pair}", response_model=APIResponse[Dict[str, Any]])
async def get_orderbook(
    connector_name: str,
    trading_pair: str,
    depth: int = Query(10, ge=1, le=100),
    app=Depends(get_hummingbot_application)
):
    """
    Get order book for a trading pair.
    """
    try:
        if connector_name not in app.markets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Connector '{connector_name}' not active"
            )

        connector = app.markets[connector_name]

        if not hasattr(connector, 'get_order_book'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Connector '{connector_name}' does not support order book"
            )

        order_book = connector.get_order_book(trading_pair)

        # Format order book
        bids = []
        asks = []

        if order_book:
            for price, amount in list(order_book.bid_entries())[:depth]:
                bids.append({"price": float(price), "amount": float(amount)})
            for price, amount in list(order_book.ask_entries())[:depth]:
                asks.append({"price": float(price), "amount": float(amount)})

        return APIResponse(data={
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.utcnow().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting orderbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))
