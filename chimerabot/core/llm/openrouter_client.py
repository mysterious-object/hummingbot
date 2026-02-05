import asyncio
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from chimerabot.core.api_throttler.async_throttler import AsyncThrottler
from chimerabot.core.llm.openrouter_constants import (
    CHAT_COMPLETIONS_ENDPOINT,
    DEFAULT_BASE_URL,
    MODELS_ENDPOINT,
    RATE_LIMITS,
    REST_CALL_RATE_LIMIT_ID,
)
from chimerabot.core.web_assistant.connections.data_types import RESTMethod
from chimerabot.core.web_assistant.web_assistants_factory import WebAssistantsFactory


class OpenRouterClient:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        http_referer: str = "",
        app_title: str = "",
        request_timeout: float = 30.0,
        models_cache_ttl: float = 3600.0,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http_referer = http_referer
        self._app_title = app_title
        self._request_timeout = request_timeout
        self._models_cache_ttl = models_cache_ttl

        self._models_cache: List[Dict[str, Any]] = []
        self._models_cache_ts: float = 0.0
        self._models_cache_lock = asyncio.Lock()

        throttler = AsyncThrottler(rate_limits=RATE_LIMITS)
        self._api_factory = WebAssistantsFactory(throttler=throttler)

    @property
    def api_key(self) -> str:
        return self._api_key

    def set_api_key(self, api_key: str):
        self._api_key = api_key

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._app_title:
            headers["X-Title"] = self._app_title
        return headers

    async def list_models(self, refresh: bool = False) -> List[Dict[str, Any]]:
        now = time.time()
        if (
            not refresh
            and self._models_cache
            and (now - self._models_cache_ts) < self._models_cache_ttl
        ):
            return self._models_cache

        async with self._models_cache_lock:
            now = time.time()
            if (
                not refresh
                and self._models_cache
                and (now - self._models_cache_ts) < self._models_cache_ttl
            ):
                return self._models_cache

            url = f"{self._base_url}{MODELS_ENDPOINT}"
            rest_assistant = await self._api_factory.get_rest_assistant()
            response = await rest_assistant.execute_request(
                url=url,
                throttler_limit_id=REST_CALL_RATE_LIMIT_ID,
                headers=self._build_headers() or None,
                timeout=self._request_timeout,
            )
            data = response.get("data")
            if data is None:
                raise IOError("OpenRouter models response missing 'data' field.")

            self._models_cache = data
            self._models_cache_ts = time.time()
            return data

    async def list_free_models(self, refresh: bool = False) -> List[Dict[str, Any]]:
        models = await self.list_models(refresh=refresh)
        return [model for model in models if self.is_free_model(model)]

    async def is_model_free(self, model_id: str, refresh: bool = False) -> bool:
        models = await self.list_models(refresh=refresh)
        for model in models:
            if model.get("id") == model_id:
                return self.is_free_model(model)
        return False

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = f"{self._base_url}{CHAT_COMPLETIONS_ENDPOINT}"
        rest_assistant = await self._api_factory.get_rest_assistant()
        response = await rest_assistant.execute_request(
            url=url,
            method=RESTMethod.POST,
            data=payload,
            throttler_limit_id=REST_CALL_RATE_LIMIT_ID,
            headers=self._build_headers() or None,
            timeout=self._request_timeout,
        )
        return response

    @classmethod
    def is_free_model(cls, model: Dict[str, Any]) -> bool:
        pricing = model.get("pricing") or {}
        return cls._price_is_zero(pricing.get("prompt")) and cls._price_is_zero(pricing.get("completion"))

    @staticmethod
    def _price_is_zero(value: Any) -> bool:
        if value is None:
            return False
        try:
            return Decimal(str(value)) == Decimal("0")
        except (InvalidOperation, ValueError, TypeError):
            return False
