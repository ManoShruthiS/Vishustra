"""Golden-table tests for the pipeline synthesizer.

Every case pins an intent to its expected pipeline (+params). These cases
double as the acceptance criteria for any future LLM steering layer: it must
produce the same shape.
"""

import pytest

from vishustra_core.kernel.synthesizer import extract_params, synthesize


GOLDEN = [
    # (intent, expected pipeline, expected params subset)
    ("summarize this", ["text_summarizer"], {}),
    ("make it polite", ["tone_converter"], {"target_tone": "polite"}),
    ("translate to hindi", ["language_translator"], {"target_language": "hi"}),
    (
        "translate to hindi and make it formal",
        ["language_translator", "tone_converter"],
        {"target_language": "hi", "target_tone": "formal"},
    ),
    (
        "clean this, remove private info, summarize to 3 lines, make it polite",
        ["profanity_filter", "pii_redactor", "tone_converter", "text_summarizer"],
        {"target_tone": "polite", "min_sentences": 3, "max_sentences": 3},
    ),
    (
        "detect sentiment and extract keywords",
        ["sentiment_analyzer", "keyword_extractor"],
        {},
    ),
    ("check if this claim is true", ["fact_checker"], {}),
    ("extract the urls and links", ["url_extractor"], {}),
    ("find numbers in the text", ["regex_matcher"], {}),
    ("embed this text", ["embedding_generator"], {}),
    ("export as json", ["json_formatter"], {"json_indent": 2}),
    ("dedupe this data", ["cache_manager"], {}),
    ("classify the intent", ["intent_classifier"], {}),
    ("convert markdown to plain text", ["markdown_parser"], {}),
    ("summarize then detect sentiment", ["text_summarizer", "sentiment_analyzer"], {}),
    ("extract phone numbers", ["regex_matcher"], {}),
]


@pytest.mark.parametrize("intent,expected_pipeline,expected_params", GOLDEN)
def test_golden_pipeline(intent, expected_pipeline, expected_params):
    plan = synthesize(intent)
    assert plan["pipeline"] == expected_pipeline, plan
    for key, value in expected_params.items():
        assert plan["params"].get(key) == value, plan["params"]


def test_reasons_align_with_pipeline():
    plan = synthesize("clean this, make it polite")
    assert len(plan["reasons"]) == len(plan["pipeline"])
    for step, reason in zip(plan["pipeline"], plan["reasons"]):
        assert reason.startswith(f"{step}:")


def test_skills_metadata_present():
    plan = synthesize("summarize this")
    assert plan["skills"]
    assert plan["skills"][0]["id"] == "text_summarizer"


def test_unsupported_intent_yields_empty_pipeline():
    plan = synthesize("hello there my friend")
    assert plan["pipeline"] == []


def test_kernel_raises_on_empty_pipeline(kernel):
    from vishustra_core.kernel.executor import VishustraKernelError

    with pytest.raises(VishustraKernelError):
        kernel.process("hello there my friend", "some text")


def test_extract_params_language_precedence():
    params = extract_params("please translate to hindi then to tamil")
    # LANGUAGE_ALIASES iteration is deterministic: 'hindi' wins over 'tamil'.
    assert params["target_language"] == "hi"