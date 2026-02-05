import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from chimerabot.client.config.config_helpers import load_client_config_map_from_file
from chimerabot.core.llm import OpenRouterClient
from chimerabot.data_feed.candles_feed.data_types import CandlesConfig
from chimerabot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a trading signal engine. Decide the direction for the next interval and "
    "return ONLY a JSON object with keys: signal (-1, 0, or 1) and confidence (0 to 1)."
)
DEFAULT_USER_PROMPT_TEMPLATE = (
    "Market features JSON:\n{features}\nReturn the JSON object only."
)


class OpenRouterLLMControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "openrouter_llm"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data (leave empty to use trading connector): ",
            "prompt_on_new": True,
        },
    )
    candles_trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data (leave empty to use trading pair): ",
            "prompt_on_new": True,
        },
    )
    interval: str = Field(
        default="5m",
        json_schema_extra={
            "prompt": "Enter the candle interval for features (e.g., 1m, 5m, 1h): ",
            "prompt_on_new": True,
        },
    )
    max_records: int = Field(
        default=60,
        ge=10,
        json_schema_extra={
            "prompt": "Enter the number of candles to use for features (min 10): ",
            "prompt_on_new": True,
        },
    )
    model_id: str = Field(
        default="",
        json_schema_extra={
            "prompt": "OpenRouter model id to use (e.g., a free-tier model id): ",
            "prompt_on_new": True,
            "is_updatable": True,
        },
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        json_schema_extra={
            "prompt": "OpenRouter temperature (0-2): ",
            "prompt_on_new": True,
            "is_updatable": True,
        },
    )
    max_tokens: int = Field(
        default=200,
        ge=32,
        json_schema_extra={
            "prompt": "OpenRouter max tokens for the response: ",
            "prompt_on_new": True,
            "is_updatable": True,
        },
    )
    inference_interval: int = Field(
        default=60,
        ge=5,
        json_schema_extra={
            "prompt": "Minimum seconds between OpenRouter calls: ",
            "prompt_on_new": True,
            "is_updatable": True,
        },
    )
    min_confidence: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "prompt": "Minimum confidence to act on a signal (0-1): ",
            "prompt_on_new": True,
            "is_updatable": True,
        },
    )
    free_models_only: bool = Field(
        default=True,
        json_schema_extra={
            "prompt": "Only allow free OpenRouter models? (True/False): ",
            "prompt_on_new": True,
            "is_updatable": True,
        },
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        json_schema_extra={
            "prompt_on_new": False,
            "is_updatable": True,
        },
    )
    user_prompt_template: str = Field(
        default=DEFAULT_USER_PROMPT_TEMPLATE,
        json_schema_extra={
            "prompt_on_new": False,
            "is_updatable": True,
        },
    )

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class OpenRouterLLMController(DirectionalTradingControllerBase):
    def __init__(self, config: OpenRouterLLMControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = self.config.max_records
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records,
            )]
        super().__init__(config, *args, **kwargs)

        self._openrouter_client, self._models_refresh_interval = self._build_openrouter_client()
        self._last_inference_ts: float = 0.0
        self._last_model_id: str = self.config.model_id
        self._model_is_free: Optional[bool] = None
        self._model_checked_ts: float = 0.0
        self._last_error: str = ""

    async def update_processed_data(self):
        if not self.config.model_id:
            self._last_error = "OpenRouter model_id is not set."
            self.processed_data["signal"] = 0
            return
        if not self._openrouter_client.api_key:
            self._last_error = "OpenRouter API key is not set."
            self.processed_data["signal"] = 0
            return

        if self._last_model_id != self.config.model_id:
            self._last_model_id = self.config.model_id
            self._model_is_free = None
            self._model_checked_ts = 0.0

        if not await self._validate_model():
            self.processed_data["signal"] = 0
            return

        now = time.time()
        if self._last_inference_ts and (now - self._last_inference_ts) < self.config.inference_interval:
            return

        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records,
        )
        if df.empty or len(df) < self.max_records:
            self._last_error = "Insufficient candle data for LLM inference."
            self.processed_data["signal"] = 0
            return

        features = self._build_features(df)
        prompt = self._build_prompt(features)
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._openrouter_client.chat_completion(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as e:
            self._last_error = f"OpenRouter request failed: {e}"
            self.logger().warning(self._last_error, exc_info=True)
            self.processed_data["signal"] = 0
            return

        decision = self._parse_decision(response)
        signal = decision.get("signal", 0)
        confidence = decision.get("confidence", 0.0)
        if confidence < self.config.min_confidence:
            signal = 0

        self.processed_data["signal"] = signal
        self.processed_data["features"] = features
        self.processed_data["decision"] = decision
        self._last_inference_ts = time.time()
        self._last_error = ""

    def to_format_status(self) -> List[str]:
        decision = self.processed_data.get("decision", {})
        lines = [
            f"Model: {self.config.model_id or 'N/A'}",
            f"Signal: {self.processed_data.get('signal', 'N/A')}",
            f"Confidence: {decision.get('confidence', 'N/A')}",
        ]
        if self._last_error:
            lines.append(f"Last error: {self._last_error}")
        return lines

    def _build_openrouter_client(self) -> Tuple[OpenRouterClient, int]:
        client_config = load_client_config_map_from_file()
        openrouter_config = client_config.openrouter
        api_key = openrouter_config.api_key.get_secret_value()
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY", "")
        client = OpenRouterClient(
            api_key=api_key,
            base_url=openrouter_config.base_url,
            http_referer=openrouter_config.http_referer,
            app_title=openrouter_config.app_title,
            request_timeout=openrouter_config.request_timeout,
            models_cache_ttl=openrouter_config.models_refresh_interval,
        )
        return client, openrouter_config.models_refresh_interval

    async def _validate_model(self) -> bool:
        if not self.config.free_models_only:
            return True

        now = time.time()
        if self._model_checked_ts and (now - self._model_checked_ts) < self._models_refresh_interval:
            return bool(self._model_is_free)

        try:
            self._model_is_free = await self._openrouter_client.is_model_free(self.config.model_id)
            self._model_checked_ts = now
        except Exception as e:
            self._last_error = f"OpenRouter model validation failed: {e}"
            self.logger().warning(self._last_error, exc_info=True)
            self._model_is_free = False

        if not self._model_is_free:
            self._last_error = (
                f"Model '{self.config.model_id}' is not free or was not found in OpenRouter listings."
            )
        return bool(self._model_is_free)

    def _build_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        closes = df["close"].astype(float)
        returns = closes.pct_change()

        features = {
            "symbol": self.config.candles_trading_pair,
            "interval": self.config.interval,
            "last_close": self._safe_float(closes.iloc[-1]),
            "return_1": self._safe_float(returns.iloc[-1]),
            "return_3": self._safe_float(self._return_over_period(closes, 3)),
            "return_5": self._safe_float(self._return_over_period(closes, 5)),
            "sma_fast": self._safe_float(self._sma(closes, 5)),
            "sma_slow": self._safe_float(self._sma(closes, 20)),
            "volatility_10": self._safe_float(returns.rolling(10).std().iloc[-1] if len(returns) >= 10 else 0.0),
        }
        return features

    def _build_prompt(self, features: Dict[str, Any]) -> str:
        features_json = json.dumps(features, separators=(",", ":"), sort_keys=True)
        return self.config.user_prompt_template.format(features=features_json)

    @staticmethod
    def _return_over_period(closes: pd.Series, periods: int) -> float:
        if len(closes) <= periods:
            return 0.0
        return (closes.iloc[-1] / closes.iloc[-1 - periods]) - 1.0

    @staticmethod
    def _sma(closes: pd.Series, window: int) -> float:
        if len(closes) < window:
            return float(closes.iloc[-1])
        return float(closes.rolling(window).mean().iloc[-1])

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            value_f = float(value)
            if math.isnan(value_f) or math.isinf(value_f):
                return 0.0
            return value_f
        except Exception:
            return 0.0

    def _parse_decision(self, response: Dict[str, Any]) -> Dict[str, Any]:
        content = ""
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            content = ""

        parsed = self._parse_json_content(content)
        signal = parsed.get("signal", 0)
        confidence = parsed.get("confidence", 0.0)

        try:
            signal = int(signal)
        except Exception:
            signal = 0
        if signal not in (-1, 0, 1):
            signal = 0

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        parsed["signal"] = signal
        parsed["confidence"] = confidence
        return parsed

    @staticmethod
    def _parse_json_content(content: str) -> Dict[str, Any]:
        if not content:
            return {"signal": 0, "confidence": 0.0}
        trimmed = content.strip()
        if trimmed.startswith("```"):
            trimmed = trimmed.strip("`")
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"signal": 0, "confidence": 0.0}
        json_str = trimmed[start:end + 1]
        try:
            return json.loads(json_str)
        except Exception:
            return {"signal": 0, "confidence": 0.0}
