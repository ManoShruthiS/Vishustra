import re
import logging
from typing import Any, Dict, Optional, Pattern

# Assuming BaseNode is located here as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# Default PII patterns and their redaction tags
# These patterns are designed to be general and can be extended or overridden.
DEFAULT_PII_PATTERNS: Dict[str, str] = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:[-.\s]?\d{1,})?\b', # Catches various phone number formats
    "ssn": r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', # US SSN format XXX-XX-XXXX
    "credit_card": r'\b(?:\d[ -]*?){13,16}\b', # 13-16 digits with optional spaces/hyphens
    "ip_address": r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
    "url": r'https?://(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?' # Basic URL pattern
}

# Redaction format string. The tag will be uppercased.
REDACTION_FORMAT = "[{tag}_REDACTED]"

class PiiRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts personally identifiable information (PII)
    from text data based on predefined or custom regular expression patterns.

    This node is crucial for ensuring privacy and compliance by preventing sensitive
    information from being exposed downstream in the LLM orchestration flow.
    """

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        """
        Initializes the PII Redactor Node with a set of PII detection patterns.

        Args:
            custom_patterns (Optional[Dict[str, str]]): An optional dictionary of
                                                       custom PII patterns. Keys are
                                                       descriptive tags (e.g., "email"),
                                                       values are regex strings.
                                                       If provided, these patterns will
                                                       *replace* the default patterns.
                                                       Use this to fine-tune or extend
                                                       PII detection.
        """
        self._compiled_patterns: Dict[str, Pattern[str]] = {}
        patterns_to_use = custom_patterns if custom_patterns is not None else DEFAULT_PII_PATTERNS

        for tag, pattern_str in patterns_to_use.items():
            try:
                self._compiled_patterns[tag] = re.compile(pattern_str, re.IGNORECASE)
            except re.error as e:
                logger.error(f"Failed to compile regex pattern for tag '{tag}': {e}. This pattern will be skipped.")
                # We log an error and skip the problematic pattern to avoid node initialization failure
                # for potentially user-provided malformed regex.

        logger.info(f"PiiRedactorNode initialized with {len(self._compiled_patterns)} active patterns for redaction.")
        if not self._compiled_patterns:
            logger.warning("No valid PII redaction patterns were loaded. This node will perform no redaction.")


    @property
    def node_name(self) -> str:
        """Returns the human-readable name of the node."""
        return "PII Redactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        If the input `data` is a string, it will be scanned against all configured
        PII patterns. Any matched PII will be replaced with a generic placeholder
        (e.g., '[EMAIL_REDACTED]').
        If `data` is not a string, it will be returned as-is after logging a warning,
        as PII redaction is primarily a text-based operation.

        Args:
            data (Any): The input data to process. Expected to be a string for effective
                        PII redaction.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. This node does not directly
                                       utilize the context, but it's part of the
                                       BaseNode interface.

        Returns:
            Any: The processed data with PII redacted if it was a string,
                 or the original data if it was not a string or if no PII was found.
        """
        if not isinstance(data, str):
            logger.warning(
                f"PiiRedactorNode received non-string data of type '{type(data).__name__}'. "
                "PII redaction can only be applied to strings. Returning original data without modification."
            )
            return data

        redacted_data = data
        total_redaction_count = 0

        if not self._compiled_patterns:
            logger.debug("PiiRedactorNode has no active patterns; returning original data.")
            return data

        for tag, pattern in self._compiled_patterns.items():
            initial_length = len(redacted_data)
            redacted_data, num_replacements = pattern.subn(REDACTION_FORMAT.format(tag=tag.upper()), redacted_data)
            
            if num_replacements > 0:
                logger.debug(f"Redacted {num_replacements} instances of '{tag}' PII.")
                total_redaction_count += num_replacements

        if total_redaction_count > 0:
            logger.info(f"Successfully redacted {total_redaction_count} instances of PII across various categories.")
        else:
            logger.debug("No PII found or redacted in the provided data string.")

        return redacted_data

