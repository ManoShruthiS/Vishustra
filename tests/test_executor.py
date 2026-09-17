"""Tests for the pipeline executor: coercion, the text lane, and error handling."""

import pytest

from vishustra_core.kernel.executor import (
    PipelineExecutor,
    VishustraKernelError,
    _coerce_for,
    _to_text,
    _wrap_statement,
)


# ---------------------------------------------------------------------------
# coercion helpers
# ---------------------------------------------------------------------------

def test_to_text_from_dict():
    assert _to_text({"sentiment": "positive", "score": 0.9}) == "sentiment: positive; score: 0.9"


def test_to_text_from_list():
    assert _to_text(["a", "b"]) == "a, b"


def test_wrap_statement():
    assert _wrap_statement("hello") == {"statement": "hello"}
    assert _wrap_statement({"statement": "x"}) == {"statement": "x"}


def test_coerce_for_str_accepting_node_coerces_dict():
    coerced = _coerce_for({"accepts": "str"}, {"a": 1})
    assert coerced == "a: 1"


def test_empty_pipeline_rejected():
    with pytest.raises(VishustraKernelError):
        PipelineExecutor({"pipeline": []})


# ---------------------------------------------------------------------------
# text lane: analyze steps read the cleaned text, not the previous analyze dict
# ---------------------------------------------------------------------------

def test_keywords_come_from_text_not_sentiment_dict(kernel):
    report = kernel.process(
        "detect sentiment and extract keywords",
        "The app crashed on my phone, truly terrible",
    )
    assert report["pipeline"] == ["sentiment_analyzer", "keyword_extractor"]
    keywords = report["result"]["final"]
    assert isinstance(keywords, list)
    assert "phone" in keywords
    assert "crashed" in keywords
    assert "sentiment" not in keywords  # prove it wasn't the dict metadata
    assert "score" not in keywords


def test_analyze_receives_sanitized_text_lane(kernel):
    # PII is redacted before sentiment runs; the sentiment result must reflect
    # the *cleaned* text lane.
    report = kernel.process(
        "remove private info and tell me sentiment",
        "This is really great, love it! Contact me at sue@x.com.",
    )
    assert report["pipeline"] == ["pii_redactor", "sentiment_analyzer"]
    assert report["result"]["final"]["sentiment"] == "positive"


# ---------------------------------------------------------------------------
# cross-contract composition
# ---------------------------------------------------------------------------

def test_double_transform_composes_to_string(kernel):
    report = kernel.process(
        "translate to hindi and make it formal",
        "Please tell the manager about the delay.",
    )
    assert report["pipeline"] == ["language_translator", "tone_converter"]
    assert isinstance(report["result"]["final"], str)


def test_trace_records_types_and_status(kernel):
    report = kernel.process("summarize this", "A long and boring sentence. Another sentence follows.")
    for entry in report["result"]["trace"]:
        assert entry["step"]
        assert entry["status"] == "ok"
        assert entry["input_type"] and entry["output_type"]


# ---------------------------------------------------------------------------
# failure handling
# ---------------------------------------------------------------------------

def test_step_failure_raises_kernel_error():
    plan = {
        "pipeline": ["text_summarizer"],
        "params": {"min_sentences": 10, "max_sentences": 1},  # invalid range
    }
    executor = PipelineExecutor(plan)
    with pytest.raises(VishustraKernelError):
        executor.run("some text")


def test_unsupported_intent_raises(kernel):
    with pytest.raises(VishustraKernelError):
        kernel.process("do something totally unknown", "content")


def test_outputs_collected_per_step(kernel):
    report = kernel.process(
        "clean this and summarize it",
        "This is terrible crap. The second sentence is longer than the first one here.",
    )
    outputs = report["result"]["outputs"]
    assert "profanity_filter" in outputs
    assert "text_summarizer" in outputs
    assert report["result"]["final"] == outputs["text_summarizer"]