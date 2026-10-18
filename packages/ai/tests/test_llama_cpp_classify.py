"""Offline tests for the llama.cpp classifier (port of upstream ``test/llama-cpp-classify.test.ts``)."""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from pi_ai.abort import AbortController
from pi_ai.api.llama_cpp_classify import (
    answer_from_probabilities,
    classify,
    label_probabilities,
    llama_server_root,
    peak_confidence,
    render_question,
)
from pi_ai.types import (
    ClassifierBoolAnswer,
    ClassifierBoolQuestion,
    ClassifierChoiceAnswer,
    ClassifierChoiceQuestion,
    ClassifierContext,
    ClassifierModel,
    ClassifierScoreAnswer,
    ClassifierScoreQuestion,
    ModelCost,
)

_server_count = itertools.count(1)


def model() -> ClassifierModel:
    """A fresh server URL per test: label tokens are cached per server and model."""
    return ClassifierModel(
        id="qwen",
        name="qwen",
        api="llama-cpp-classify",
        provider="llama.cpp",
        base_url=f"http://llama-{next(_server_count)}.test:8080/v1",
        input=["text"],
        cost=ModelCost(),
        context_window=32768,
    )


@dataclass
class RecordedRequest:
    url: str
    body: dict[str, Any]
    headers: httpx.Headers


def char_tokens(content: str) -> list[int]:
    """Token IDs: one per character, the character code."""
    return [ord(char) for char in content]


@dataclass
class FakeServer:
    #: Log-probabilities of the next token by token text, in rank order.
    next: Callable[[str, int], dict[str, float]] | None = None
    template: Callable[[list[dict[str, str]]], str] | None = None
    tokenize: Callable[[str], list[int]] | None = None
    requests: list[RecordedRequest] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(RecordedRequest(str(request.url), body, request.headers))
        path = request.url.path
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": (self.tokenize or char_tokens)(str(body["content"]))})
        if path == "/apply-template":
            messages = body["messages"]
            prompt = (
                self.template(messages)
                if self.template
                else "".join(f"<|{message['role']}|>\n{message['content']}\n" for message in messages)
                + "<|assistant|>\n"
            )
            return httpx.Response(200, json={"prompt": prompt})
        if path == "/completion":
            ranked = self.next(str(body["prompt"]), int(body["n_probs"])) if self.next else {"A": -0.1, "B": -2.5}
            top_logprobs = [
                {"id": ord(token[0]), "token": token, "bytes": [], "logprob": logprob}
                for token, logprob in ranked.items()
            ]
            return httpx.Response(
                200,
                json={
                    "content": "A",
                    "completion_probabilities": [{"id": 65, "token": "A", "top_logprobs": top_logprobs}],
                },
            )
        return httpx.Response(404, text="not found")

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def answer_by_prompt(prompt: str, _depth: int) -> dict[str, float]:
    """Completion log-probabilities for bool (Yes/No), score and letter labels."""
    if "Answer Yes or No." in prompt:
        return {"Y": -0.05, "N": -3}
    if "Answer with one level number." in prompt:
        return {"2": -0.2, "1": -1.8, "0": -4}
    return {"B": -0.3, "A": -1.5, "C": -3}


CONTEXT = ClassifierContext(
    state={"message": "Help! My payouts have been failing for 3 days."},
    questions={
        "team": ClassifierChoiceQuestion(
            "Which team should handle this?",
            {"billing": "Payments and refunds", "technical": "Bugs and outages", "sales": ""},
        ),
        "urgent": ClassifierBoolQuestion(
            "Does this convey urgency?", {"true": "The user needs help soon", "false": "No time pressure"}
        ),
        "severity": ClassifierScoreQuestion("How severe is this?", ["low", "medium", "high"]),
    },
)

PICK = {"pick": ClassifierChoiceQuestion("Pick one", {"a": "", "b": ""})}


def word_tokens(content: str) -> list[int]:
    """Maps multi-character labels to single tokens, as a real vocabulary would."""
    words = {"Yes": 89, "No": 78}
    tokens: list[int] = []
    for part in re.split(r"(\n)", content):
        if part == "":
            continue
        tokens.extend([words[part]] if part in words else char_tokens(part))
    return tokens


def completion_depths(requests: list[RecordedRequest]) -> list[int]:
    return [int(request.body["n_probs"]) for request in requests if request.url.endswith("/completion")]


async def test_answers_choice_bool_and_score_questions_from_label_log_probabilities():
    server = FakeServer(next=answer_by_prompt, tokenize=word_tokens)
    classifier_model = model()

    result = await classify(classifier_model, CONTEXT, {"api_key": "local", "http_client": server.client()})

    assert result.error_message is None
    assert result.stop_reason == "stop"
    choice = label_probabilities([-1.5, -0.3, -3], 1)
    assert result.answers["team"] == ClassifierChoiceAnswer(
        "technical",
        {"billing": choice[0], "technical": choice[1], "sales": choice[2]},
        peak_confidence(choice),
    )
    assert result.answers["urgent"] == ClassifierBoolAnswer(label_probabilities([-0.05, -3], 1)[0])
    levels = label_probabilities([-4, -1.8, -0.2], 1)
    assert result.answers["severity"] == ClassifierScoreAnswer(levels[1] + 2 * levels[2], peak_confidence(levels))

    root = classifier_model.base_url.removesuffix("/v1")
    for request in server.requests:
        assert request.url.startswith(f"{root}/")
        assert request.body["model"] == "qwen"
        assert request.headers["authorization"] == "Bearer local"
    completion = next(request for request in server.requests if request.url.endswith("/completion"))
    assert completion.body == {
        "model": "qwen",
        "prompt": completion.body["prompt"],
        "n_predict": 1,
        "n_probs": 256,
        "post_sampling_probs": False,
        "cache_prompt": True,
        "temperature": 0,
    }
    template = next(request for request in server.requests if request.url.endswith("/apply-template"))
    assert template.body["chat_template_kwargs"] == {"enable_thinking": False}
    assert template.body["messages"][0]["role"] == "system"
    tokenize = next(request for request in server.requests if request.url.endswith("/tokenize"))
    assert tokenize.body["add_special"] is False
    assert tokenize.body["parse_special"] is False


def test_repeats_the_state_around_all_questions_and_ends_with_this_questions_labels():
    rendered = render_question(CONTEXT, "team")
    state = 'State:\n{\n "message": "Help! My payouts have been failing for 3 days."\n}'
    assert rendered.labels == ["A", "B", "C"]
    assert rendered.keys == ["billing", "technical", "sales"]
    assert rendered.content == "\n".join(
        [
            state,
            "",
            "Task: answer each of the following questions about the state.",
            "",
            "Question: Which team should handle this?",
            "",
            "Options:",
            "- billing: Payments and refunds",
            "- technical: Bugs and outages",
            "- sales",
            "",
            "Question: Does this convey urgency?",
            "",
            "Yes means: The user needs help soon",
            "No means: No time pressure",
            "",
            "Question: How severe is this?",
            "",
            "Levels:",
            "0. low",
            "1. medium",
            "2. high",
            "",
            state,
            "",
            "Question: Which team should handle this?",
            "",
            "Options:",
            "A. billing: Payments and refunds",
            "B. technical: Bugs and outages",
            "C. sales",
            "",
            "Answer with one letter.",
        ]
    )


def test_renders_a_single_question_overview_and_unknown_questions():
    rendered = render_question(ClassifierContext(state={}, questions=PICK), "pick")
    assert rendered.content.startswith("State:\n{}\n\nTask: answer the following question about the state.")
    with pytest.raises(ValueError, match="Unknown question: missing"):
        render_question(CONTEXT, "missing")


def test_shares_everything_before_the_final_question_across_the_questions_of_a_request():
    def prefix(question_id: str) -> str:
        content = render_question(CONTEXT, question_id).content
        return content[: content.rindex("Question:")]

    assert prefix("urgent") == prefix("team")
    assert prefix("severity") == prefix("team")
    assert render_question(CONTEXT, "urgent").content.endswith("No means: No time pressure\n\nAnswer Yes or No.")
    assert render_question(CONTEXT, "severity").content.endswith("2. high\n\nAnswer with one level number.")


async def test_divides_label_log_probabilities_by_the_temperature():
    server = FakeServer(next=lambda _prompt, _depth: {"A": -0.1, "B": -2.5})

    result = await classify(
        model(), ClassifierContext(state={}, questions=PICK), {"http_client": server.client(), "temperature": 2}
    )

    expected = label_probabilities([-0.1 / 2, -2.5 / 2], 1)
    assert result.answers["pick"].probabilities == {"a": expected[0], "b": expected[1]}
    assert label_probabilities([-0.1, -2.5], 2) == expected
    # No API key: no authorization header.
    assert all("authorization" not in request.headers for request in server.requests)


async def test_rejects_non_positive_temperatures_before_sending_requests():
    server = FakeServer()

    result = await classify(model(), CONTEXT, {"http_client": server.client(), "temperature": 0})

    assert result.stop_reason == "error"
    assert "Temperature must be a positive number, got 0" in (result.error_message or "")
    assert server.requests == []


async def test_retries_deeper_readouts_when_a_label_is_missing_and_fails_without_inventing_zeros():
    deep = FakeServer(next=lambda _prompt, depth: {"A": -0.1} if depth < 4096 else {"A": -0.1, "B": -9})
    recovered = await classify(model(), ClassifierContext(state={}, questions=PICK), {"http_client": deep.client()})
    assert recovered.stop_reason == "stop"
    assert completion_depths(deep.requests) == [256, 4096]

    never = FakeServer(next=lambda _prompt, _depth: {"A": -0.1})
    failed = await classify(model(), ClassifierContext(state={}, questions=PICK), {"http_client": never.client()})
    assert failed.stop_reason == "error"
    assert failed.answers == {}
    assert "did not rank labels B for pick within the top 32768 tokens" in (failed.error_message or "")
    assert completion_depths(never.requests) == [256, 4096, 32768]


async def test_rejects_answers_that_underflow_for_every_label():
    server = FakeServer(next=lambda _prompt, _depth: {"A": -1e30, "B": -1e30})

    result = await classify(model(), ClassifierContext(state={}, questions=PICK), {"http_client": server.client()})

    assert result.error_message == "qwen gave no probability to any answer label for pick"


async def test_closes_a_reasoning_block_the_template_leaves_open():
    server = FakeServer(template=lambda _messages: "<|assistant|>\n<think>")

    await classify(
        model(),
        ClassifierContext(state={}, questions={"pick": ClassifierChoiceQuestion("Pick", {"a": "", "b": ""})}),
        {"http_client": server.client()},
    )

    completion = next(request for request in server.requests if request.url.endswith("/completion"))
    assert completion.body["prompt"] == "<|assistant|>\n<think></think>"


async def test_reads_labels_in_reply_position_and_rejects_labels_that_are_not_one_token():
    # A tokenizer that merges a newline with a following letter falls back to the label alone.
    merging = FakeServer(
        tokenize=lambda content: [1000] if content.startswith("\n") and len(content) > 1 else char_tokens(content)
    )
    merged = await classify(
        model(),
        ClassifierContext(state={}, questions={"pick": ClassifierChoiceQuestion("Pick", {"a": "", "b": ""})}),
        {"http_client": merging.client()},
    )
    assert merged.stop_reason == "stop"

    # The default fake tokenizer splits "Yes" into three tokens.
    split = FakeServer()
    result = await classify(
        model(),
        ClassifierContext(state={}, questions={"ok": ClassifierBoolQuestion("OK?", {"true": "", "false": ""})}),
        {"http_client": split.client()},
    )
    assert result.stop_reason == "error"
    assert 'Label "Yes" is not a single token for qwen' in (result.error_message or "")


async def test_rejects_labels_that_share_a_token():
    server = FakeServer(tokenize=lambda content: [7] if content != "\n" else [10])

    result = await classify(model(), ClassifierContext(state={}, questions=PICK), {"http_client": server.client()})

    assert result.error_message == "Labels share a token for qwen: A, B"


async def test_caches_label_tokens_per_server_and_model():
    classifier_model = model()
    server = FakeServer()
    context = ClassifierContext(state={}, questions={"pick": ClassifierChoiceQuestion("Pick", {"a": "", "b": ""})})

    await classify(classifier_model, context, {"http_client": server.client()})
    first_tokenizations = sum(1 for request in server.requests if request.url.endswith("/tokenize"))
    await classify(classifier_model, context, {"http_client": server.client()})

    assert first_tokenizations > 0
    assert sum(1 for request in server.requests if request.url.endswith("/tokenize")) == first_tokenizations


async def test_validates_option_counts_before_sending_requests():
    server = FakeServer()
    criteria = {f"option{index}": "" for index in range(63)}

    too_many = await classify(
        model(),
        ClassifierContext(state={}, questions={"pick": ClassifierChoiceQuestion("Pick", criteria)}),
        {"http_client": server.client()},
    )
    too_few = await classify(
        model(),
        ClassifierContext(state={}, questions={"rate": ClassifierScoreQuestion("Rate", ["only"])}),
        {"http_client": server.client()},
    )

    assert "A choice question needs 2 to 62 options, got 63" in (too_many.error_message or "")
    assert "A score question needs 2 to 10 levels, got 1" in (too_few.error_message or "")
    assert server.requests == []


async def test_passes_completion_payloads_and_responses_through_the_request_hooks():
    server = FakeServer()
    payloads: list[Any] = []
    statuses: list[int] = []

    def on_payload(payload: Any, _model: Any) -> Any:
        payloads.append(payload)
        return {**payload, "id_slot": 1}

    await classify(
        model(),
        ClassifierContext(state={}, questions={"pick": ClassifierChoiceQuestion("Pick", {"a": "", "b": ""})}),
        {
            "http_client": server.client(),
            "on_payload": on_payload,
            "on_response": lambda response, _model: statuses.append(response.status),
        },
    )

    assert len(payloads) == 1
    assert payloads[0]["n_predict"] == 1
    assert statuses == [200]
    completion = next(request for request in server.requests if request.url.endswith("/completion"))
    assert completion.body["id_slot"] == 1


async def test_reports_server_errors_and_cancellation(mock_server):
    mock_server.add(httpx.Response(400, text='{"error":{"message":"context overflow"}}'))
    failing = await classify(model(), CONTEXT, {"max_retries": 0, "http_client": mock_server.client()})
    assert failing.stop_reason == "error"
    assert "llama.cpp error (400)" in (failing.error_message or "")
    assert "context overflow" in (failing.error_message or "")

    controller = AbortController()
    controller.abort()
    server = FakeServer()
    aborted = await classify(model(), CONTEXT, {"signal": controller.signal, "http_client": server.client()})
    assert aborted.stop_reason == "aborted"
    assert aborted.error_message == "Request aborted"
    assert server.requests == []


async def test_rejects_models_for_other_classifier_apis():
    server = FakeServer()
    other = model()
    other.api = "typesafe-system-one"

    result = await classify(other, CONTEXT, {"http_client": server.client()})

    assert "Unsupported classifier API: typesafe-system-one" in (result.error_message or "")
    assert server.requests == []


def test_derives_the_server_root_from_openai_compatible_base_urls():
    assert llama_server_root("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080"
    assert llama_server_root("https://example.com/prefix/v1") == "https://example.com/prefix"
    assert llama_server_root("http://127.0.0.1:8080") == "http://127.0.0.1:8080"


def test_computes_typesafes_confidence_and_expected_scores():
    assert peak_confidence([0.89, 0.06, 0.05]) == pytest.approx(0.835)
    assert peak_confidence([0.5, 0.5]) == 0
    assert peak_confidence([1, 0, 0]) == 1
    assert answer_from_probabilities(
        ClassifierScoreQuestion("", ["a", "b", "c"]), ["0", "1", "2"], [0.2, 0.3, 0.5]
    ) == ClassifierScoreAnswer(1.3, peak_confidence([0.2, 0.3, 0.5]))
