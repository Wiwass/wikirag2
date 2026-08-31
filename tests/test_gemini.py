"""Offline tests for the Gemini client (wikirag/llm/gemini.py).

No network: every google-genai client is a stub installed over the module's
single construction point (_make_genai_client), and backoff/carousel waits
go through _sleep, which is replaced by a recorder.
"""

from typing import Any, Sequence

import pytest

from wikirag.llm import gemini
from wikirag.llm.client import AllKeysExhausted, Usage
from wikirag.llm.meter import MeteredClient


class FakeUsageMetadata:
    def __init__(
        self,
        prompt: int = 0,
        output: int = 0,
        total: int = 0,
        cached: int = 0,
    ) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = output
        self.total_token_count = total
        self.cached_content_token_count = cached


class FakeFinishReason:
    """Stands in for the SDK enum, which carries the name on `.name`."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeCandidate:
    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = FakeFinishReason(finish_reason)


class FakeResponse:
    def __init__(
        self,
        text: str = "ok",
        finish_reason: str = "STOP",
        usage: FakeUsageMetadata | None = None,
    ) -> None:
        self.text = text
        self.candidates = [FakeCandidate(finish_reason)]
        self.usage_metadata = usage


class ApiError(Exception):
    """google-genai errors expose the HTTP status on `.code`."""

    def __init__(self, code: int | None = None, message: str = "") -> None:
        super().__init__(message or f"error {code}")
        self.code = code


class FakeModels:
    def __init__(self, fake: "FakeGenai", api_key: str) -> None:
        self._fake = fake
        self._api_key = api_key

    def generate_content(self, *, model: str, contents: Any, config: Any) -> FakeResponse:
        self._fake.calls.setdefault(self._api_key, []).append(
            {"model": model, "contents": contents, "config": config}
        )
        script = self._fake.scripts.get(self._api_key)
        outcome = script.pop(0) if script else self._fake.default
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeGenaiClient:
    def __init__(self, fake: "FakeGenai", api_key: str) -> None:
        self.api_key = api_key
        self.models = FakeModels(fake, api_key)


class FakeGenai:
    """Installs itself as gemini._make_genai_client and records per-key use.

    `scripts` maps an api key to the outcomes its client returns or raises,
    in order; once a key's script runs out its `default` repeats. Scripts and
    call logs live on the fake, not on the client object, so they survive the
    client being rebuilt when the pool rotates back onto the same key.
    """

    def __init__(
        self,
        scripts: dict[str, Sequence[Any]] | None = None,
        default: Any = None,
    ) -> None:
        self.scripts: dict[str, list[Any]] = {k: list(v) for k, v in (scripts or {}).items()}
        self.default = default if default is not None else FakeResponse()
        self.constructed: list[str] = []
        self.calls: dict[str, list[dict[str, Any]]] = {}

    def __call__(self, api_key: str) -> FakeGenaiClient:
        self.constructed.append(api_key)
        return FakeGenaiClient(self, api_key)

    def calls_for(self, api_key: str) -> list[dict[str, Any]]:
        return self.calls.get(api_key, [])


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(gemini, "_sleep", recorded.append)
    return recorded


def install(monkeypatch: pytest.MonkeyPatch, fake: FakeGenai) -> FakeGenai:
    monkeypatch.setattr(gemini, "_make_genai_client", fake)
    return fake


# --- success path -----------------------------------------------------------


def test_success_maps_text_usage_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(
        monkeypatch,
        FakeGenai(
            default=FakeResponse(
                text="  hello  ",
                finish_reason="STOP",
                usage=FakeUsageMetadata(prompt=1000, output=500, total=1500, cached=200),
            )
        ),
    )
    client = gemini.GeminiClient("gemini-2.5-flash", ["k1"])

    result = client.generate(system="sys", user="usr")

    assert result.text == "hello"
    assert result.truncated is False
    assert result.usage == Usage(
        prompt_tokens=1000, output_tokens=500, total_tokens=1500, cached_tokens=200
    )
    assert result.cost_estimate == pytest.approx((1000 * 0.30 + 500 * 2.50) / 1_000_000)
    assert fake.constructed == ["k1"]

    call = fake.calls_for("k1")[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"] == "usr"
    assert call["config"].system_instruction == "sys"
    assert call["config"].max_output_tokens == 8192
    assert call["config"].response_mime_type is None


def test_missing_usage_metadata_yields_zero_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeGenai(default=FakeResponse(text="x", usage=None)))
    result = gemini.GeminiClient("gemini-2.5-flash", ["k1"]).generate(system="s", user="u")
    assert result.usage == Usage()
    assert result.cost_estimate == 0.0


def test_json_schema_reaches_the_request_config(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch, FakeGenai())
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}

    gemini.GeminiClient("gemini-2.5-flash", ["k1"]).generate(
        system="s", user="u", json_schema=schema, max_output_tokens=256
    )

    config = fake.calls_for("k1")[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema == schema
    assert config.max_output_tokens == 256


def test_max_tokens_finish_reason_sets_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeGenai(default=FakeResponse(text="cut", finish_reason="MAX_TOKENS")))
    result = gemini.GeminiClient("gemini-2.5-flash", ["k1"]).generate(system="s", user="u")
    assert result.truncated is True
    assert result.text == "cut"


def test_model_attribute_is_exposed_for_metering(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeGenai())
    client = gemini.GeminiClient("gemini-2.5-flash-lite", ["k1"])
    assert client.model == "gemini-2.5-flash-lite"

    recorded: list[dict[str, Any]] = []

    class Store:
        def record_llm_call(self, **row: Any) -> None:
            recorded.append(row)

        def get_llm_calls(self) -> list[dict]:
            return list(recorded)

    metered = MeteredClient(client, store=Store(), stage="enrich", role="enrich")
    metered.generate(system="s", user="u")
    assert recorded[0]["model"] == "gemini-2.5-flash-lite"


# --- retries and key rotation ----------------------------------------------


def test_transient_errors_back_off_then_succeed(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    fake = install(
        monkeypatch,
        FakeGenai(
            scripts={"k1": [ApiError(500, "internal"), ApiError(503, "unavailable")]},
            default=FakeResponse(text="third time"),
        ),
    )
    result = gemini.GeminiClient("gemini-2.5-flash", ["k1"]).generate(system="s", user="u")

    assert result.text == "third time"
    assert sleeps == [2.0, 4.0]  # base_delay * 2**(attempt-1)
    assert len(fake.calls_for("k1")) == 3
    assert fake.constructed == ["k1"]  # same key, same client


def test_transient_errors_propagate_after_the_last_retry(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    install(monkeypatch, FakeGenai(default=ApiError(500, "internal")))
    client = gemini.GeminiClient("gemini-2.5-flash", ["k1"], max_retries_per_key=2)

    with pytest.raises(ApiError):
        client.generate(system="s", user="u")
    assert len(sleeps) == 1


def test_bad_request_is_not_retried(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    fake = install(monkeypatch, FakeGenai(default=ApiError(400, "malformed")))
    with pytest.raises(ApiError):
        gemini.GeminiClient("gemini-2.5-flash", ["k1", "k2"]).generate(system="s", user="u")
    assert len(fake.calls_for("k1")) == 1
    assert sleeps == []


def test_quota_error_rotates_to_the_next_key(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    fake = install(
        monkeypatch,
        FakeGenai(
            scripts={"k1": [ApiError(429, "RESOURCE_EXHAUSTED. Please retry in 12.5s")] * 2},
            default=FakeResponse(text="from k2"),
        ),
    )
    client = gemini.GeminiClient("gemini-2.5-flash", ["k1", "k2"], max_retries_per_key=2)

    result = client.generate(system="s", user="u")

    assert result.text == "from k2"
    assert fake.constructed == ["k1", "k2"]
    assert len(fake.calls_for("k1")) == 2 and len(fake.calls_for("k2")) == 1
    assert sleeps == [17.5]  # the delay Google asked for, plus margin


def test_broken_key_rotates_and_is_never_used_again(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    fake = install(
        monkeypatch,
        FakeGenai(
            scripts={
                "k1": [ApiError(401, "UNAUTHENTICATED: service account disabled")],
                "k2": [ApiError(429, "RESOURCE_EXHAUSTED")],
            },
            default=FakeResponse(text="from k3"),
        ),
    )
    client = gemini.GeminiClient("gemini-2.5-flash", ["k1", "k2", "k3"], max_retries_per_key=1)

    assert client.generate(system="s", user="u").text == "from k3"
    assert len(fake.calls_for("k1")) == 1  # no retry on a broken key
    assert sleeps == []

    # A second call restarts at the broken key's successor, not at k1.
    assert client.generate(system="s", user="u").text == "from k3"
    assert len(fake.calls_for("k1")) == 1


def test_all_keys_exhausted_raises(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    install(monkeypatch, FakeGenai(default=ApiError(429, "RESOURCE_EXHAUSTED")))
    client = gemini.GeminiClient("gemini-2.5-flash", ["k1", "k2"], max_retries_per_key=1)

    with pytest.raises(AllKeysExhausted) as excinfo:
        client.generate(system="s", user="u")
    assert isinstance(excinfo.value.__cause__, ApiError)


def test_carousel_waits_restarts_the_pool_and_is_capped(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    fake = install(monkeypatch, FakeGenai(default=ApiError(429, "RESOURCE_EXHAUSTED")))
    client = gemini.GeminiClient(
        "gemini-2.5-flash",
        ["k1", "k2"],
        carousel=True,
        carousel_wait_seconds=70.0,
        max_carousel_rounds=2,
        max_retries_per_key=1,
    )

    with pytest.raises(AllKeysExhausted):
        client.generate(system="s", user="u")

    assert sleeps == [70.0, 70.0]  # two restarts, then it gives up
    assert fake.constructed == ["k1", "k2", "k1", "k2", "k1", "k2"]


def test_carousel_succeeds_after_a_restart(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    fake = install(
        monkeypatch,
        FakeGenai(
            scripts={"k1": [ApiError(429, "RESOURCE_EXHAUSTED")]},
            default=FakeResponse(text="quota came back"),
        ),
    )
    client = gemini.GeminiClient(
        "gemini-2.5-flash", ["k1"], carousel=True, max_retries_per_key=1
    )

    assert client.generate(system="s", user="u").text == "quota came back"
    assert sleeps == [70.0]
    assert fake.constructed == ["k1", "k1"]


def test_carousel_gives_up_when_every_key_is_broken(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    install(monkeypatch, FakeGenai(default=ApiError(403, "PERMISSION_DENIED")))
    client = gemini.GeminiClient(
        "gemini-2.5-flash", ["k1", "k2"], carousel=True, max_retries_per_key=1
    )

    with pytest.raises(AllKeysExhausted):
        client.generate(system="s", user="u")
    assert sleeps == []  # waiting cannot un-break a key


# --- key loading ------------------------------------------------------------


def test_load_api_keys_reads_txt_files_in_name_order(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(gemini.SINGLE_KEY_ENV, raising=False)
    (tmp_path / "key_b.txt").write_text("kb1\n\nkb2\n", encoding="utf-8")
    (tmp_path / "key_a.txt").write_text("ka1\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignored\n", encoding="utf-8")
    monkeypatch.setenv(gemini.KEYS_DIR_ENV, str(tmp_path))

    assert gemini.load_api_keys() == ["ka1", "kb1", "kb2"]
    assert gemini.load_api_keys(tmp_path) == ["ka1", "kb1", "kb2"]


def test_load_api_keys_deduplicates(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.txt").write_text("same\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("same\n", encoding="utf-8")
    monkeypatch.setenv(gemini.KEYS_DIR_ENV, str(tmp_path))
    assert gemini.load_api_keys() == ["same"]


def test_load_api_keys_falls_back_to_the_single_env_key(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(gemini.KEYS_DIR_ENV, str(tmp_path / "absent"))
    monkeypatch.setenv(gemini.SINGLE_KEY_ENV, " env-key ")
    assert gemini.load_api_keys() == ["env-key"]


def test_client_loads_keys_lazily_at_first_call(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, FakeGenai())
    monkeypatch.setenv(gemini.KEYS_DIR_ENV, str(tmp_path))
    client = gemini.GeminiClient("gemini-2.5-flash")  # no keys yet: must not raise
    (tmp_path / "pool.txt").write_text("lazy-key\n", encoding="utf-8")

    client.generate(system="s", user="u")
    assert fake.constructed == ["lazy-key"]


def test_no_keys_raises_a_clear_error_at_first_call(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeGenai())
    monkeypatch.setenv(gemini.KEYS_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(gemini.SINGLE_KEY_ENV, raising=False)
    client = gemini.GeminiClient("gemini-2.5-flash")

    with pytest.raises(RuntimeError, match=gemini.SINGLE_KEY_ENV):
        client.generate(system="s", user="u")


# --- price table ------------------------------------------------------------


def test_estimate_cost_uses_the_price_table() -> None:
    usage = Usage(prompt_tokens=2_000_000, output_tokens=1_000_000)
    assert gemini.estimate_cost("gemini-2.5-pro", usage) == pytest.approx(2 * 1.25 + 10.00)
    assert gemini.estimate_cost("gemini-2.0-flash-lite", usage) == pytest.approx(2 * 0.075 + 0.30)


def test_estimate_cost_matches_the_longest_family_prefix() -> None:
    usage = Usage(prompt_tokens=1_000_000)
    assert gemini.estimate_cost("gemini-2.5-flash-lite-preview-09-2025", usage) == pytest.approx(0.10)
    assert gemini.estimate_cost("gemini-2.5-flash-preview-09-2025", usage) == pytest.approx(0.30)


def test_estimate_cost_is_zero_for_an_unknown_model() -> None:
    assert gemini.estimate_cost("some-local-model", Usage(prompt_tokens=10_000)) == 0.0
