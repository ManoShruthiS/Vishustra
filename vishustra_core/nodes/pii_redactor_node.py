import logging
import re
from typing import Any, Dict, List, Optional, Union

# Assuming BaseNode is located here as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node responsible for identifying and redacting
    Personally Identifiable Information (PII) from text data.

    This node traverses input data (strings, dictionaries, lists) and applies
    configured regular expression patterns to replace detected PII with
    a specified redaction placeholder.
    """

    # Default PII patterns with associated redaction placeholders.
    # These are designed to be general and can be overridden during initialization.
    DEFAULT_PII_PATTERNS: List[Dict[str, Union[str, re.Pattern[str]]]] = [
        {
            "name": "Email Address",
            "pattern": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "redaction": "[EMAIL_REDACTED]"
        },
        {
            "name": "Phone Number (US-like)",
            "pattern": re.compile(r'\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'),
            "redaction": "[PHONE_REDACTED]"
        },
        {
            "name": "Credit Card Number (basic detection)",
            "pattern": re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})\b'),
            "redaction": "[CREDIT_CARD_REDACTED]"
        },
        {
            "name": "IP Address (IPv4)",
            "pattern": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            "redaction": "[IP_REDACTED]"
        }
    ]

    def __init__(self, pii_patterns: Optional[List[Dict[str, Union[str, re.Pattern[str]]]]] = None):
        """
        Initializes the PIIRedactorNode with an optional list of PII patterns.

        Args:
            pii_patterns: A list of dictionaries, where each dictionary
                          must contain:
                          - 'name': A descriptive string for the pattern.
                          - 'pattern': A compiled regular expression (`re.Pattern`).
                          - 'redaction': The string to replace matched PII with.
                          If None, a set of default patterns will be used.
        """
        self._pii_patterns = pii_patterns if pii_patterns is not None else self.DEFAULT_PII_PATTERNS
        logger.info("PIIRedactorNode initialized with %d redaction patterns.", len(self._pii_patterns))
        for p_info in self._pii_patterns:
            if not isinstance(p_info.get("pattern"), re.Pattern):
                logger.error("Pattern for '%s' is not a compiled regex. This pattern will be skipped.", p_info.get("name", "Unnamed pattern"))
                continue
            logger.debug("Pattern loaded: '%s' - Regex: %s", p_info.get("name"), p_info["pattern"].pattern)

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PIIRedactorNode"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured PII redaction patterns to a single string.

        Args:
            text: The input string to redact.

        Returns:
            The string with identified PII replaced by redaction placeholders.
        """
        redacted_text = text
        for pii_info in self._pii_patterns:
            pattern = pii_info.get("pattern")
            redaction = pii_info.get("redaction")
            name = pii_info.get("name", "Unnamed pattern")

            if not isinstance(pattern, re.Pattern):
                continue # Skip invalid patterns

            try:
                original_text_before_pattern = redacted_text
                redacted_text = pattern.sub(redaction, redacted_text)
                if original_text_before_pattern != redacted_text:
                    logger.debug("Redacted '%s' in string using pattern '%s'.", name, pattern.pattern)
            except Exception as e:
                logger.error("Error applying PII redaction pattern '%s' (%s): %s", name, pattern.pattern, e)
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        This method supports:
        - Strings: Direct application of PII redaction patterns.
        - Dictionaries: Recursive traversal and redaction of string values.
        - Lists/Tuples: Recursive traversal and redaction of items.
        - Other types: Returned unchanged.

        Args:
            data: The input data, which can be a string, dict, list, tuple, or other.
            context: A dictionary containing contextual information for the processing.
                     (Not directly used for redaction logic in this node, but passed
                     through for consistency with the BaseNode interface and potential
                     future extensions.)

        Returns:
            The input data with identified PII redacted. The data structure
            is preserved.
        """
        logger.debug("PIIRedactorNode received data of type %s for processing.", type(data))

        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            redacted_data = {}
            for key, value in data.items():
                redacted_data[key] = self.process(value, context)  # Recursive call
            return redacted_data
        elif isinstance(data, (list, tuple)):
            # Process each item in the list/tuple recursively, maintaining type
            redacted_collection = [self.process(item, context) for item in data]
            return type(data)(redacted_collection) # Reconstruct as original type (list or tuple)
        else:
            logger.debug("Data of type %s is not a string, dict, list, or tuple. Returning unchanged.", type(data))
            return data