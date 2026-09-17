"""VISHUSTRA KERNEL: the skill library.

Every node in vishustra_core.nodes is described here as a composable *skill*
that the synthesizer can reason about: what it accepts, what it returns, what
context parameters it needs, and which natural-language words imply it.

Adding a skill here (hand-written or via Ayan) makes it available to the
pipeline synthesizer. This registry is the single source of truth for the
Kernel; the nodes themselves remain plain BaseNode implementations.
"""

import json
import os

_RUNTIME_SKILLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime_skills.json")


def _load_runtime_skills() -> dict:
    """Merge skills registered at runtime (e.g. by the Ayan worker) into SKILLS."""
    runtime = {}
    if os.path.exists(_RUNTIME_SKILLS_PATH):
        try:
            with open(_RUNTIME_SKILLS_PATH, "r", encoding="utf-8") as f:
                runtime = json.load(f)
        except (OSError, ValueError):
            runtime = {}
    return runtime


SKILLS = {
    "profanity_filter": {
        "id": "profanity_filter",
        "node_name": "ProfanityFilter",
        "module": "vishustra_core.nodes.profanity_filter",
        "class": "ProfanityFilterNode",
        "category": "sanitize",
        "output": "str",
        "accepts": "str",
        "params": {},
        "description": "Replaces profane/offensive words with masked symbols.",
        "keywords": ["clean", "offensive", "bad words", "profanity", "abusive", "swear", "vulgar", "filter"],
    },
    "pii_redactor": {
        "id": "pii_redactor",
        "node_name": "PII Redactor",
        "module": "vishustra_core.nodes.pii_redactor",
        "class": "PIIRedactorNode",
        "category": "sanitize",
        "output": "str",
        "accepts": "str",
        "params": {},
        "description": "Anonymizes emails/phone numbers and other personally identifiable info.",
        "keywords": ["redact", "pii", "anonymize", "remove personal", "remove private", "private info", "privacy", "hide email", "hide phone", "sensitive"],
    },
    "url_extractor": {
        "id": "url_extractor",
        "node_name": "URL Extractor",
        "module": "vishustra_core.nodes.url_extractor",
        "class": "URLExtractorNode",
        "category": "sanitize",
        "output": "list",
        "accepts": "str",
        "params": {},
        "description": "Extracts every unique URL present in the text.",
        "keywords": ["url", "link", "links", "extract url", "web address", "hyperlink"],
    },
    "markdown_parser": {
        "id": "markdown_parser",
        "node_name": "MarkdownParserNode",
        "module": "vishustra_core.nodes.markdown_parser",
        "class": "MarkdownParserNode",
        "category": "sanitize",
        "output": "str",
        "accepts": "str",
        "params": {},
        "description": "Converts Markdown (or HTML) formatting to plain readable text.",
        "keywords": ["markdown", "md", "formatting", "strip formatting", "plain text", "convert markdown", "clean html"],
    },
    "language_translator": {
        "id": "language_translator",
        "node_name": "LanguageTranslator",
        "module": "vishustra_core.nodes.language_translator",
        "class": "LanguageTranslatorNode",
        "category": "transform",
        "output": "str",
        "accepts": "str",
        "params": {"target_language": "en"},
        "description": "Translates text into a target language.",
        "keywords": ["translate", "translation", "to hindi", "to tamil", "to french", "to spanish", "to english", "in hindi", "into hindi", "language"],
    },
    "tone_converter": {
        "id": "tone_converter",
        "node_name": "ToneConverter",
        "module": "vishustra_core.nodes.tone_converter",
        "class": "ToneConverter",
        "category": "transform",
        "output": "str",
        "accepts": "str",
        "params": {"target_tone": "professional"},
        "description": "Rewrites text with a target tone (polite, formal, friendly, professional, sarcastic).",
        "keywords": ["polite", "tone", "formal", "friendly", "respectful", "professional", "nice", "soften", "kind", "sarcastic", "manner"],
    },
    "text_summarizer": {
        "id": "text_summarizer",
        "node_name": "TextSummarizer",
        "module": "vishustra_core.nodes.text_summarizer",
        "class": "TextSummarizerNode",
        "category": "transform",
        "output": "str",
        "accepts": "str",
        "params": {"min_sentences": 2, "max_sentences": 5, "summary_ratio": 0.2},
        "description": "Summarizes long text into a short extractive summary.",
        "keywords": ["summarize", "summary", "summarise", "short", "brief", "condense", "shorten", "tl;dr", "tl dr", "digest", "key points"],
    },
    "sentiment_analyzer": {
        "id": "sentiment_analyzer",
        "node_name": "SentimentAnalyzer",
        "module": "vishustra_core.nodes.sentiment_analyzer",
        "class": "SentimentAnalyzer",
        "category": "analyze",
        "output": "dict",
        "accepts": "str",
        "params": {},
        "description": "Detects the sentiment (positive/negative/neutral) and a score.",
        "keywords": ["sentiment", "emotion", "positive", "negative", "angry", "happy", "mood", "feeling", "opinion"],
    },
    "intent_classifier": {
        "id": "intent_classifier",
        "node_name": "IntentClassifierNode",
        "module": "vishustra_core.nodes.intent_classifier_node",
        "class": "IntentClassifierNode",
        "category": "analyze",
        "output": "str",
        "accepts": "str",
        "params": {},
        "description": "Classifies the overall intent/category of a message.",
        "keywords": ["intent", "classify", "category", "type", "purpose", "which type", "what kind"],
    },
    "keyword_extractor": {
        "id": "keyword_extractor",
        "node_name": "KeywordExtractorNode",
        "module": "vishustra_core.nodes.keyword_extractor",
        "class": "KeywordExtractorNode",
        "category": "analyze",
        "output": "list",
        "accepts": "str",
        "params": {},
        "description": "Extracts the most meaningful keywords/topics from text.",
        "keywords": ["keyword", "keywords", "topics", "important words", "extract words", "theme", "subjects"],
    },
    "regex_matcher": {
        "id": "regex_matcher",
        "node_name": "RegexMatcher",
        "module": "vishustra_core.nodes.regex_matcher",
        "class": "RegexMatcherNode",
        "category": "analyze",
        "output": "list",
        "accepts": "str",
        "params": {"return_all_matches": True, "group_index": 0},
        "description": "Finds text matching a regex pattern (numbers, codes, ids, etc.).",
        "keywords": ["regex", "pattern", "match pattern", "numbers", "codes", "find ids", "extract numbers", "validate format"],
    },
    "embedding_generator": {
        "id": "embedding_generator",
        "node_name": "EmbeddingsGenerator",
        "module": "vishustra_core.nodes.embedding_generator",
        "class": "EmbeddingsGeneratorNode",
        "category": "analyze",
        "output": "list",
        "accepts": "str",
        "params": {"embedding_dimension": 8},
        "description": "Embeds text into a numeric vector (for similarity/clustering).",
        "keywords": ["embedding", "embed", "vector", "similarity", "similar", "vectors", "cluster"],
    },
    "fact_checker": {
        "id": "fact_checker",
        "node_name": "FactCheckerNode",
        "module": "vishustra_core.nodes.fact_checker_node",
        "class": "FactCheckerNode",
        "category": "analyze",
        "output": "dict",
        "accepts": "dict_statement",
        "params": {},
        "description": "Checks a factual claim against a knowledge base (VERIFIED/REFUTED/UNVERIFIED).",
        "keywords": ["fact", "truth", "true", "verify", "claim", "check facts", "is it true", "real?"],
    },
    "data_validator": {
        "id": "data_validator",
        "node_name": "DataValidator",
        "module": "vishustra_core.nodes.data_validator",
        "class": "DataValidator",
        "category": "validate",
        "output": "raw",
        "accepts": "any",
        "params": {"validation_schema": None},
        "description": "Validates structured data against a schema.",
        "keywords": ["validate", "validation", "schema", "check format", "conform"],
    },
    "schema_validator": {
        "id": "schema_validator",
        "node_name": "Schema Validator Node",
        "module": "vishustra_core.nodes.data_schema_validator_node",
        "class": "SchemaValidatorNode",
        "category": "validate",
        "output": "raw",
        "accepts": "any",
        "params": {},
        "description": "Validates data structure integrity.",
        "keywords": ["schema check", "structural", "integrity", "structure"],
    },
    "json_formatter": {
        "id": "json_formatter",
        "node_name": "JSONFormatter",
        "module": "vishustra_core.nodes.json_formatter_node",
        "class": "JSONFormatterNode",
        "category": "format",
        "output": "str",
        "accepts": "any",
        "params": {"json_indent": 2},
        "description": "Serializes any data into a formatted JSON string.",
        "keywords": ["json", "format", "structured", "export", "serialize"],
    },
    "cache_manager": {
        "id": "cache_manager",
        "node_name": "CacheManager",
        "module": "vishustra_core.nodes.cache_manager",
        "class": "CacheManager",
        "category": "store",
        "output": "raw",
        "accepts": "any",
        "params": {"cache_action": "set", "cache_key": "default"},
        "description": "Stores/dedupes data through an in-memory cache.",
        "keywords": ["cache", "dedupe", "dedup", "duplicate", "store", "memoize"],
    },
}

_loaded_runtime = _load_runtime_skills()
if _loaded_runtime:
    SKILLS.update(_loaded_runtime)

CATEGORY_ORDER = {"sanitize": 1, "transform": 2, "analyze": 3, "validate": 4, "store": 5, "format": 6}

LANGUAGE_ALIASES = {
    "hindi": "hi", "tamil": "ta", "english": "en", "french": "fr", "spanish": "es",
    "german": "de", "telugu": "te", "malayalam": "ml", "kannada": "kn", "bengali": "bn",
    "marathi": "mr", "urdu": "ur", "chennai": "ta", "tamil nadu": "ta",
}


def skill_keywords(skill_id: str):
    return SKILLS[skill_id]["keywords"]