import threading
from decimal import Decimal
from typing import TYPE_CHECKING

from chimerabot.connector.utils import split_hb_trading_pair, validate_trading_pair
from chimerabot.core.rate_oracle.rate_oracle import RateOracle
from chimerabot.core.utils.async_utils import safe_ensure_future
from chimerabot.exceptions import OracleRateUnavailable

s_float_0 = float(0)
s_decimal_0 = Decimal("0")

if TYPE_CHECKING:
    from chimerabot.client.chimerabot_application import ChimeraBotApplication  # noqa: F401


class RateCommand:
    def rate(self,  # type: ChimeraBotApplication
             pair: str,
             token: str
             ):
        if threading.current_thread() != threading.main_thread():
            self.ev_loop.call_soon_threadsafe(self.trades)
            return
        if pair:
            safe_ensure_future(self.show_rate(pair))
        elif token:
            safe_ensure_future(self.show_token_value(token))

    async def show_rate(self,  # type: ChimeraBotApplication
                        pair: str,
                        ):
        if not validate_trading_pair(pair):
            self.notify(f"Invalid trading pair {pair}")
        else:
            try:
                msg = await self.oracle_rate_msg(pair)
            except OracleRateUnavailable:
                msg = "Rate is not available."
            self.notify(msg)

    async def oracle_rate_msg(self,  # type: ChimeraBotApplication
                              pair: str):
        if not validate_trading_pair(pair):
            self.notify(f"Invalid trading pair {pair}")
        else:
            pair = pair.upper().strip('\"').strip("'")
            rate = await RateOracle.get_instance().rate_async(pair)
            if rate is None:
                raise OracleRateUnavailable
            base, quote = split_hb_trading_pair(pair)
            return f"Source: {RateOracle.get_instance().source.name}\n1 {base} = {rate} {quote}"

    async def show_token_value(self,  # type: ChimeraBotApplication
                               token: str
                               ):
        if "-" in token:
            self.notify(f"Expected a single token but got a pair {token}")
        else:
            self.notify(f"Source: {RateOracle.get_instance().source.name}")
            rate = await RateOracle.get_instance().get_rate(base_token=token)
            if rate is None:
                self.notify("Rate is not available.")
                return
            global_token = self.client_config_map.global_token.global_token_name
            token_symbol = self.client_config_map.global_token.global_token_symbol
            self.notify(f"1 {token} = {token_symbol} {rate} {global_token}")
