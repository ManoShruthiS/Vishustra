"""Contract tests for every BaseNode in vishustra_core.nodes.

Covers the happy path, the heterogeneous contracts (str->str, str->dict,
str->list, dict->dict, any->numeric vector), and the important edge cases
(non-string input, empty input, missing required context params).
"""

import pytest


# ---------------------------------------------------------------------------
# sanitize lane
# ---------------------------------------------------------------------------

def test_profanity_filter_masks_words():
    from vishustra_core.nodes.profanity_filter import ProfanityFilterNode

    node = ProfanityFilterNode()
    assert "This is ***" in node.process("This is SHIT service.", {})


def test_profanity_filter_passes_non_strings_through():
    from vishustra_core.nodes.profanity_filter import ProfanityFilterNode

    node = ProfanityFilterNode()
    payload = {"text": "hello"}
    assert node.process(payload, {}) is payload


def test_pii_redactor_hides_email_and_phone():
    from vishustra_core.nodes.pii_redactor import PIIRedactorNode

    node = PIIRedactorNode()
    out = node.process("Reach me at john@x.com or +1-555-123-4567 today.", {})
    assert "john@x.com" not in out
    assert "555-123-4567" not in out
    assert "[EMAIL_REDACTED]" in out
    assert "[PHONE_REDACTED]" in out


def test_pii_redactor_passes_non_strings_through():
    from vishustra_core.nodes.pii_redactor import PIIRedactorNode

    node = PIIRedactorNode()
    assert node.process(12345, {}) == 12345


def test_url_extractor_returns_sorted_unique_urls():
    from vishustra_core.nodes.url_extractor import URLExtractorNode

    node = URLExtractorNode()
    urls = node.process(
        "See https://example.com/report and www.news.org, again https://example.com/report.", {}
    )
    assert urls == ["https://example.com/report", "www.news.org"]


def test_url_extractor_handles_empty_text():
    from vishustra_core.nodes.url_extractor import URLExtractorNode

    node = URLExtractorNode()
    assert node.process("no links here", {}) == []


def test_markdown_parser_strips_heading_and_bold():
    from vishustra_core.nodes.markdown_parser import MarkdownParserNode

    node = MarkdownParserNode()
    out = node.process("# Title\nSome **bold** content.", {})
    assert "Title" in out
    assert "bold" in out
    assert "**" not in out


def test_markdown_parser_rejects_non_string():
    from vishustra_core.nodes.markdown_parser import MarkdownParserNode

    node = MarkdownParserNode()
    with pytest.raises(TypeError):
        node.process(["not", "text"], {})


# ---------------------------------------------------------------------------
# transform lane
# ---------------------------------------------------------------------------

def test_language_translator_requires_target_language():
    from vishustra_core.nodes.language_translator import LanguageTranslatorNode

    node = LanguageTranslatorNode()
    with pytest.raises(ValueError):
        node.process("hello", {})
    out = node.process("hello", {"target_language": "hi"})
    assert "translated to hi" in out


def test_tone_converter_formal_template():
    from vishustra_core.nodes.tone_converter import ToneConverter

    node = ToneConverter()
    out = node.process("Please review this.", {"target_tone": "formal"})
    assert "Regarding the matter at hand" in out


def test_tone_converter_requires_target_tone():
    from vishustra_core.nodes.tone_converter import ToneConverter

    node = ToneConverter()
    with pytest.raises(ValueError):
        node.process("hello", {})


def test_tone_converter_unknown_tone_returns_original():
    from vishustra_core.nodes.tone_converter import ToneConverter

    node = ToneConverter()
    assert node.process("unchanged", {"target_tone": "whatever"}) == "unchanged"


def test_text_summarizer_returns_fewer_sentences():
    from vishustra_core.nodes.text_summarizer import TextSummarizerNode

    node = TextSummarizerNode()
    long_text = (
        "First sentence introduces the topic. Second sentence gives a detail. "
        "Third sentence continues with more. Fourth sentence adds depth. "
        "Fifth sentence wraps things up."
    )
    summary = node.process(long_text, {})
    assert summary
    assert len(summary) <= len(long_text)


def test_text_summarizer_empty_input():
    from vishustra_core.nodes.text_summarizer import TextSummarizerNode

    node = TextSummarizerNode()
    assert node.process("   ", {}) == ""


def test_text_summarizer_rejects_non_string():
    from vishustra_core.nodes.text_summarizer import TextSummarizerNode

    node = TextSummarizerNode()
    with pytest.raises(TypeError):
        node.process(42, {})


def test_embedding_generator_requires_dimension():
    from vishustra_core.nodes.embedding_generator import EmbeddingsGeneratorNode

    node = EmbeddingsGeneratorNode()
    with pytest.raises(ValueError):
        node.process("hello", {})
    vec = node.process("hello", {"embedding_dimension": 4})
    assert len(vec) == 4
    assert all(isinstance(x, float) for x in vec)


def test_embedding_generator_list_input():
    from vishustra_core.nodes.embedding_generator import EmbeddingsGeneratorNode

    node = EmbeddingsGeneratorNode()
    vecs = node.process(["a", "b"], {"embedding_dimension": 3})
    assert len(vecs) == 2
    assert all(len(v) == 3 for v in vecs)


# ---------------------------------------------------------------------------
# analyze lane
# ---------------------------------------------------------------------------

def test_sentiment_analyzer_contract():
    from vishustra_core.nodes.sentiment_analyzer import SentimentAnalyzer

    node = SentimentAnalyzer()
    result = node.process("This is absolutely amazing fantastic!", {})
    assert isinstance(result, dict)
    assert result["sentiment"] == "positive"
    assert -1.0 <= result["score"] <= 1.0


def test_sentiment_analyzer_rejects_non_string():
    from vishustra_core.nodes.sentiment_analyzer import SentimentAnalyzer

    node = SentimentAnalyzer()
    with pytest.raises(TypeError):
        node.process(7, {})


def test_keyword_extractor_contract():
    from vishustra_core.nodes.keyword_extractor import KeywordExtractorNode

    node = KeywordExtractorNode()
    keywords = node.process("The app crashed repeatedly on my phone today.", {})
    assert isinstance(keywords, list)
    assert "phone" in keywords
    assert "the" not in keywords


def test_keyword_extractor_empty_input():
    from vishustra_core.nodes.keyword_extractor import KeywordExtractorNode

    node = KeywordExtractorNode()
    assert node.process("   ", {}) == []


def test_regex_matcher_returns_list_of_matches():
    from vishustra_core.nodes.regex_matcher import RegexMatcherNode

    node = RegexMatcherNode(pattern=r"\d{3}", return_all_matches=True)
    assert node.process("call 123 or 456", {}) == ["123", "456"]


def test_regex_matcher_no_match_returns_empty_list():
    from vishustra_core.nodes.regex_matcher import RegexMatcherNode

    node = RegexMatcherNode(pattern=r"\d{3}", return_all_matches=True)
    assert node.process("no digits at all", {}) == []


def test_intent_classifier_returns_intent():
    from vishustra_core.nodes.intent_classifier_node import IntentClassifierNode

    node = IntentClassifierNode()
    assert node.process("please help me", {}) == "request"
    assert node.process("xylophone banana", {}) == "unknown_intent"


# ---------------------------------------------------------------------------
# fact-checking (dict -> dict)
# ---------------------------------------------------------------------------

def test_fact_checker_verifies_known_statement():
    from vishustra_core.nodes.fact_checker_node import FactCheckerNode

    node = FactCheckerNode()
    result = node.process({"statement": "The capital of France is Paris."}, {})
    assert result["status"] == "VERIFIED"
    assert result["is_fact_checked"] is True


def test_fact_checker_unknown_statement_unverified():
    from vishustra_core.nodes.fact_checker_node import FactCheckerNode

    node = FactCheckerNode()
    result = node.process({"statement": "Some made up claim here."}, {})
    assert result["status"] == "UNVERIFIED"


def test_fact_checker_rejects_bad_shape():
    from vishustra_core.nodes.fact_checker_node import FactCheckerNode

    node = FactCheckerNode()
    with pytest.raises(ValueError):
        node.process("The capital of France is Paris.", {})


# ---------------------------------------------------------------------------
# validation / format / store lane
# ---------------------------------------------------------------------------

def test_data_validator_passes_without_schema():
    from vishustra_core.nodes.data_validator import DataValidator

    node = DataValidator()
    assert node.process({"anything": 1}, {}) == {"anything": 1}


def test_data_validator_enforces_schema():
    from vishustra_core.nodes.data_validator import DataValidator

    node = DataValidator()
    schema = {"name": {"type": str, "required": True, "min_length": 3}}
    assert node.process({"name": "alice"}, {"validation_schema": schema}) == {"name": "alice"}
    with pytest.raises(ValueError):
        node.process({"name": "x"}, {"validation_schema": schema})


def test_schema_validator_node_checks_types():
    from vishustra_core.nodes.data_schema_validator_node import SchemaValidatorNode

    node = SchemaValidatorNode()
    cfg = {"required_fields": ["name"], "field_types": {"name": str}}
    assert node.process({"name": "alice"}, {"validation_config": cfg})
    with pytest.raises(ValueError):
        node.process({"name": 42}, {"validation_config": cfg})


def test_json_formatter_pretty_prints():
    from vishustra_core.nodes.json_formatter_node import JSONFormatterNode

    node = JSONFormatterNode()
    out = node.process({"a": 1}, {"json_indent": 2})
    assert '"a"' in out
    assert "1" in out


def test_json_formatter_handles_plain_string():
    from vishustra_core.nodes.json_formatter_node import JSONFormatterNode

    node = JSONFormatterNode()
    assert node.process("hello", {}) == '"hello"'


def test_cache_manager_set_and_get():
    from vishustra_core.nodes.cache_manager import CacheManager

    node = CacheManager()
    store = {}
    ctx = {"cache_store": store, "cache_key": "k", "cache_action": "set"}
    assert node.process("value", ctx) == "value"
    ctx["cache_action"] = "get"
    assert node.process(None, ctx) == "value"


def test_cache_manager_clear_all():
    from vishustra_core.nodes.cache_manager import CacheManager

    node = CacheManager()
    store = {"a": 1}
    ctx = {"cache_store": store, "cache_action": "clear_all"}
    assert node.process(None, ctx) is None
    assert store == {}