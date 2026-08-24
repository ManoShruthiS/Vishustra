import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node designed to redact Personally Identifiable Information (PII)
    from text data using configurable regular expressions.

    This node is capable of redacting common PII patterns such as email addresses,
    phone numbers, and US Social Security Numbers. It provides a flexible
    mechanism to customize or extend the default PII detection patterns and
    their corresponding replacement strings.

    Supported input data types for redaction are:
    - `str`: The string itself is redacted.
    - `list[str]`: Each string item within the list is redacted.
    - `dict[str, str]`: String values within the dictionary are redacted;
      non-string values are passed through unmodified with a warning.
    """

    DEFAULT_PII_PATTERNS = {
        "email": {
            "regex": r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            "replacement": "[EMAIL_REDACTED]"
        },
        "phone": {
            "regex": r'\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b',
            "replacement": "[PHONE_REDACTED]"
        },
        "ssn": {
            "regex": r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b',  # US Social Security Number pattern
            "replacement": "[SSN_REDACTED]"
        },
        "ip_address": {
            "regex": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "replacement": "[IP_REDACTED]"
        },
        # Additional common PII patterns can be added here
        # Example: Credit Card numbers (simplified, might need more specific validation)
        # "credit_card": {
        #     "regex": r'\b(?:\d[ -]*?){13,16}\b',
        #     "replacement": "[CC_REDACTED]"
        # },
    }

    def __init__(self, pii_patterns: Dict[str, Dict[str, str]] = None):
        """
        Initializes the PIIRedactorNode with specific PII patterns.

        Args:
            pii_patterns: An optional dictionary where keys represent PII categories
                          (e.g., "email", "phone") and values are dictionaries.
                          Each inner dictionary must contain a "regex" string and
                          a "replacement" string. If provided, these patterns
                          will override or extend the node's default patterns.
        """
        self._pii_patterns = self.DEFAULT_PII_PATTERNS.copy()
        if pii_patterns:
            self._pii_patterns.update(pii_patterns)
        
        # Compile all regex patterns upon initialization for improved performance
        self._compiled_patterns = [
            (re.compile(pattern_def["regex"]), pattern_def["replacement"])
            for pattern_def in self._pii_patterns.values()
        ]
        logger.debug(f"PIIRedactorNode initialized with {len(self._compiled_patterns)} patterns.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured redaction patterns to a single string.

        Args:
            text: The input string to be redacted.

        Returns:
            The string with all identified PII redacted.
        """
        redacted_text = text
        for pattern, replacement in self._compiled_patterns:
            redacted_text = pattern.sub(replacement, redacted_text)
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to detect and redact PII.

        The method supports `str`, `list[str]`, and `dict[str, str]` as input types.
        For lists, all items must be strings. For dictionaries, only string values
        are redacted. Unsupported data types will result in a `TypeError`.

        Args:
            data: The input data, expected to be a string, a list of strings,
                  or a dictionary with string values.
            context: A dictionary containing contextual information relevant
                     to the current processing operation.

        Returns:
            The data with PII redacted, maintaining its original structure.

        Raises:
            TypeError: If the input data type is not supported for redaction,
                       or if a list contains non-string items.
        """
        logger.debug(f"PIIRedactorNode received data of type: {type(data).__name__}.")

        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, list):
            redacted_list: List[str] = []
            for item in data:
                if isinstance(item, str):
                    redacted_list.append(self._redact_string(item))
                else:
                    error_msg = (
                        f"PIIRedactorNode encountered a non-string item (type: '{type(item).__name__}') "
                        f"within a list. All list items must be strings for redaction."
                    )
                    logger.error(error_msg)
                    raise TypeError(error_msg)
            return redacted_list
        elif isinstance(data, dict):
            redacted_dict: Dict[str, Any] = {}
            for key, value in data.items():
                if isinstance(value, str):
                    redacted_dict[key] = self._redact_string(value)
                else:
                    logger.warning(
                        f"PIIRedactorNode: Dictionary value for key '{key}' is of type "
                        f"'{type(value).__name__}', not a string. Skipping redaction for this value."
                    )
                    redacted_dict[key] = value  # Retain non-string values as-is
            return redacted_dict
        else:
            error_msg = (
                f"PIIRedactorNode received unsupported data type '{type(data).__name__}'. "
                "This node only processes strings, lists of strings, or dictionaries with string values."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
