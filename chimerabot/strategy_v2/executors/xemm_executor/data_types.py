from decimal import Decimal
from typing import Literal

from chimerabot.core.data_type.common import TradeType
from chimerabot.strategy_v2.executors.data_types import ConnectorPair, ExecutorConfigBase


class XEMMExecutorConfig(ExecutorConfigBase):
    type: Literal["xemm_executor"] = "xemm_executor"
    buying_market: ConnectorPair
    selling_market: ConnectorPair
    maker_side: TradeType
    order_amount: Decimal
    min_profitability: Decimal
    target_profitability: Decimal
    max_profitability: Decimal
