from chimerabot.core.api_throttler.data_types import RateLimit

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MODELS_ENDPOINT = "/models"
CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"

REST_CALL_RATE_LIMIT_ID = "openrouter_rest"
RATE_LIMITS = [RateLimit(limit_id=REST_CALL_RATE_LIMIT_ID, limit=5, time_interval=1)]
