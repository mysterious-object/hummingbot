from chimerabot.core.llm.openrouter_client import OpenRouterClient


def test_is_free_model_true():
    model = {"pricing": {"prompt": "0", "completion": "0"}}
    assert OpenRouterClient.is_free_model(model) is True


def test_is_free_model_false_for_paid_completion():
    model = {"pricing": {"prompt": "0", "completion": "0.0001"}}
    assert OpenRouterClient.is_free_model(model) is False


def test_is_free_model_false_for_missing_pricing():
    model = {"pricing": {}}
    assert OpenRouterClient.is_free_model(model) is False
