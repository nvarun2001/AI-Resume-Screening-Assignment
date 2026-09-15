import pytest
from pydantic import BaseModel

from resume_screener.config import Settings
from resume_screener.llm_adapter import LLMClient, LLMError


class Sample(BaseModel):
    name: str
    score: int


def test_returns_validated_model():
    def transport(system, user, schema):
        return {"name": "Asha", "score": 42}

    client = LLMClient(transport=transport)
    result = client.extract_structured("sys", "user", Sample)
    assert result.name == "Asha"
    assert result.score == 42


def test_retries_then_succeeds():
    calls = {"n": 0}

    def transport(system, user, schema):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("transient")
        return {"name": "Bob", "score": 10}

    client = LLMClient(transport=transport)
    result = client.extract_structured("sys", "user", Sample)
    assert result.score == 10
    assert calls["n"] == 2


def test_raises_after_two_failures():
    def transport(system, user, schema):
        raise ValueError("always fails")

    client = LLMClient(transport=transport)
    with pytest.raises(LLMError):
        client.extract_structured("sys", "user", Sample)


def test_malformed_output_raises_llm_error():
    def transport(system, user, schema):
        return {"name": "missing score"}  # fails schema validation

    client = LLMClient(transport=transport)
    with pytest.raises(LLMError):
        client.extract_structured("sys", "user", Sample)


def test_schema_json_passed_to_transport():
    seen = {}

    def transport(system, user, schema):
        seen["schema"] = schema
        return {"name": "x", "score": 1}

    LLMClient(transport=transport).extract_structured("sys", "user", Sample)
    assert "properties" in seen["schema"]
    assert set(seen["schema"]["properties"]) == {"name", "score"}


class FakeProviderError(Exception):
    """Stand-in for an SDK-specific error such as anthropic.BadRequestError."""


def test_provider_sdk_error_becomes_llm_error():
    # A provider error must be caught and surfaced as LLMError so the batch can
    # degrade gracefully instead of crashing.
    def transport(system, user, schema):
        raise FakeProviderError("credit balance too low")

    client = LLMClient(transport=transport)
    with pytest.raises(LLMError):
        client.extract_structured("sys", "user", Sample)


def test_unconfigured_without_transport_raises():
    settings = Settings(llm_provider="", llm_model="", llm_api_key="")
    client = LLMClient(settings=settings)
    with pytest.raises(LLMError):
        client.extract_structured("sys", "user", Sample)


def test_unsupported_provider_raises():
    settings = Settings(llm_provider="mystery", llm_model="m", llm_api_key="k")
    client = LLMClient(settings=settings)
    with pytest.raises(LLMError):
        client.extract_structured("sys", "user", Sample)
