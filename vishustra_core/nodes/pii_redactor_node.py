import logging
import re
from typing import Any, Dict, List, Pattern, Union

# Assuming `BaseNode` is located at `vishustra_core/nodes/base_node.py`
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts Personally Identifiable Information (PII)
    from text data. It uses a configurable set of regular expressions to identify
    and replace sensitive information with a placeholder.

    Supports redaction within strings, and recursively within string values
    of dictionaries and lists.
    """

    _DEFAULT_REDACTION_PATTERNS: Dict[str, str] = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone_us": r'\b(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b', # Covers (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX.XXX.XXXX
        "ssn_us": r'\b\d{3}-\d{2}-\d{4}\b',
        "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
        "url": r'https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/[a-zA-Z0-9]+\.[^\s]{2,}|[a-zA-Z0-9]+\.[^\s]{2,}',
        "credit_card": r'\b(?:\d{4}[-.\s]?){3}\d{4}\b', # Basic pattern, can be more specific
    }
    """Default PII patterns provided by the node."""

    _DEFAULT_REDACTION_PLACEHOLDER: str = "[REDACTED_PII]"
    """The default string used to replace redacted PII."""

    def __init__(
        self,
        patterns_to_enable: Union[List[str], None] = None,
        custom_patterns: Union[Dict[str, str], None] = None,
        redaction_placeholder: Union[str, None] = None
    ) -> None:
        """
        Initializes the PIIRedactorNode with specific redaction rules.

        Args:
            patterns_to_enable: A list of keys from `_DEFAULT_REDACTION_PATTERNS`
                                to apply. If `None` or empty, all default patterns are applied.
            custom_patterns: A dictionary of `{name: regex_string}` for additional
                             custom PII patterns. These are added to the selected
                             default patterns.
            redaction_placeholder: The string to replace redacted PII with.
                                   Defaults to `"[REDACTED_PII]"`.
        """
        self._compiled_patterns: List[Pattern[str]] = []
        self._redaction_placeholder = redaction_placeholder or self._DEFAULT_REDACTION_PLACEHOLDER

        effective_patterns: Dict[str, str] = {}

        if not patterns_to_enable: # None or empty list
            effective_patterns.update(self._DEFAULT_REDACTION_PATTERNS)
            logger.debug(f"PIIRedactorNode initialized to redact all default PII types.")
        else:
            for pattern_key in patterns_to_enable:
                if pattern_key in self._DEFAULT_REDACTION_PATTERNS:
                    effective_patterns[pattern_key] = self._DEFAULT_REDACTION_PATTERNS[pattern_key]
                else:
                    logger.warning(
                        f"Attempted to enable unknown default PII pattern key: '{pattern_key}'. Skipping."
                    )
            logger.debug(f"PIIRedactorNode initialized with specific default PII types: {list(effective_patterns.keys())}.")

        if custom_patterns:
            for name, pattern in custom_patterns.items():
                if not isinstance(pattern, str):
                    logger.error(f"Custom pattern '{name}' is not a string regex. Skipping.")
                    continue
                effective_patterns[name] = pattern
            logger.debug(f"PIIRedactorNode initialized with custom patterns: {list(custom_patterns.keys())}.")

        for name, pattern_str in effective_patterns.items():
            try:
                self._compiled_patterns.append(re.compile(pattern_str, re.IGNORECASE)) # Ignore case for broader matching
            except re.error as e:
                logger.error(f"Failed to compile regex pattern '{name}': {e}. This pattern will not be used.")

        if not self._compiled_patterns:
            logger.warning("PIIRedactorNode initialized with no active redaction patterns. It will not redact any PII.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PIIRedactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        Args:
            data: The input data to process. Expected to be a string,
                  or a dictionary/list containing strings.
            context: A dictionary containing contextual information for the processing.
                     Not directly used for redaction logic but available to the node.

        Returns:
            The processed data with PII redacted. If the input data type is not
            a string, dictionary, or list, it is returned as is without modification.
        """
        if not self._compiled_patterns:
            logger.debug("PIIRedactorNode has no active patterns. Returning data unmodified.")
            return data

        return self._redact_data_recursively(data)

    def _redact_data_recursively(self, data: Any) -> Any:
        """
        Helper method to apply PII redaction recursively to various data structures.
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            return {k: self._redact_data_recursively(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._redact_data_recursively(item) for item in data]
        else:
            logger.debug(
                f"PIIRedactorNode received non-string/dict/list data type for redaction: "
                f"{type(data)}. Returning as is."
            )
            return data

    def _redact_string(self, text: str) -> str:
        """
        Helper method to apply all compiled redaction patterns to a string.
        """
        redacted_text = text
        for pattern in self._compiled_patterns:
            matches = pattern.findall(redacted_text)
            if matches:
                # Log a maximum of 5 matches to avoid excessive log output for very long texts
                log_matches = matches[:5]
                logger.debug(
                    f"Redacting PII using pattern: '{pattern.pattern}'. Found {len(matches)} occurrences. "
                    f"Example matches: {log_matches}..." if len(matches) > 5 else f"Example matches: {log_matches}."
                )
                redacted_text = pattern.sub(self._redaction_placeholder, redacted_text)
        return redacted_text