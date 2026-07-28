import logging
import re
from typing import Any, Dict, Union, List, Tuple

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts personally identifiable information (PII)
    from input data. It uses regular expressions to identify and replace common PII
    patterns such as email addresses, phone numbers, and credit card numbers.

    The node is designed to recursively process strings within dictionaries and lists,
    ensuring comprehensive redaction across complex data structures.
    """

    # Default regex patterns for common PII and their respective redaction masks.
    # These are illustrative and can be extended or overridden at initialization.
    _DEFAULT_PII_PATTERNS: List[Tuple[str, str]] = [
        # Email addresses: e.g., user@example.com
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),
        # US Phone numbers: e.g., (123) 456-7890, 123-456-7890, 123.456.7890
        (r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]"),
        # Credit Card Numbers: basic 13-16 digit patterns, not fully robust but indicative
        (r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|6(?:011|5\d{2})\d{12}|3[47]\d{13}|3(?:0[0-5]|[68]\d)\d{11}|(?:2131|1800|35\d{3})\d{11})\b", "[CC_REDACTED]"),
        # IP Addresses: e.g., 192.168.1.1
        (r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "[IP_REDACTED]"),
        # Social Security Numbers (US, simple pattern: XXX-XX-XXXX)
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]")
    ]

    def __init__(self, patterns: Union[List[Tuple[str, str]], None] = None):
        """
        Initializes the PII Redactor Node with custom or default redaction patterns.

        Args:
            patterns (list[tuple[str, str]], optional): A list of (regex_pattern_string, redaction_mask_string) tuples.
                                                        If None, the node uses a set of default PII patterns.
                                                        Patterns are compiled during initialization for efficiency.
        """
        self._patterns_config = patterns if patterns is not None else self._DEFAULT_PII_PATTERNS
        self._compiled_patterns = []

        for pattern_str, mask_str in self._patterns_config:
            try:
                compiled_regex = re.compile(pattern_str)
                self._compiled_patterns.append((compiled_regex, mask_str))
                logger.debug(f"Compiled PII redaction pattern: '{pattern_str}'")
            except re.error as e:
                logger.error(f"Failed to compile regex pattern '{pattern_str}': {e}. This pattern will be skipped.")
                # It's generally better to skip a malformed pattern than to halt the node,
                # especially if patterns might be user-defined or come from configuration.

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PIIRedactorNode"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured PII redaction patterns to a given string.

        Args:
            text (str): The string to redact.

        Returns:
            str: The string with identified PII replaced by their respective masks.
        """
        redacted_text = text
        for compiled_pattern, mask_str in self._compiled_patterns:
            try:
                redacted_text = compiled_pattern.sub(mask_str, redacted_text)
            except Exception as e:
                logger.warning(f"Error during redaction with pattern '{compiled_pattern.pattern}': {e}. Skipping this pattern for current text.")
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, redacting identified PII.

        This method supports various data types:
        - If `data` is a string, PII patterns within the string are redacted.
        - If `data` is a dictionary, string values (and values within nested dictionaries/lists)
          are recursively redacted. Keys are not affected.
        - If `data` is a list, string elements (and elements within nested dictionaries/lists)
          are recursively redacted.
        - For any other data type (e.g., int, float, boolean), the data is returned as-is
          without modification.

        Args:
            data (Any): The input data to be processed for PII redaction.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing pipeline. Not directly used
                                       by this node for redaction logic, but available
                                       for future extensions (e.g., dynamic pattern updates).

        Returns:
            Any: The processed data with PII redacted. The structure of the data
                 remains the same as the input.
        """
        if data is None:
            logger.debug("Received None data for PII redaction. Returning None.")
            return None

        if isinstance(data, str):
            logger.debug("Redacting PII from string data.")
            return self._redact_string(data)
        elif isinstance(data, dict):
            logger.debug("Recursively redacting PII from dictionary values.")
            redacted_dict = {}
            for key, value in data.items():
                redacted_dict[key] = self.process(value, context)  # Recursive call for values
            return redacted_dict
        elif isinstance(data, list):
            logger.debug("Recursively redacting PII from list elements.")
            redacted_list = [self.process(item, context) for item in data]  # Recursive call for elements
            return redacted_list
        else:
            logger.debug(f"Data type {type(data).__name__} is not a string, dict, or list. Returning data as-is.")
            return data