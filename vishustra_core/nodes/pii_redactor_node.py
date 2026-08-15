import logging
import re
from typing import Any, Dict, Union

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PiiRedactorNode(BaseNode):
    """
    A Vishustra processing node responsible for identifying and redacting Personally
    Identifiable Information (PII) from input data.

    It supports redaction within strings, dictionaries (recursively), and lists of data.
    """

    # Pre-compiled regex patterns for common PII types
    # These can be extended or made configurable via context in a more advanced version.
    _REDACTION_PATTERNS = {
        "email": (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[REDACTED_EMAIL]'),
        "phone_number": (re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[REDACTED_PHONE]'),
        "credit_card": (re.compile(r'\b(?:\d{4}[- ]){3}\d{4}\b'), '[REDACTED_CREDIT_CARD]'), # Simplified pattern
        "social_security": (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[REDACTED_SSN]') # US SSN format
    }

    def __init__(self) -> None:
        """
        Initializes the PII Redactor Node.
        """
        logger.debug("PiiRedactorNode initialized, ready for PII redaction tasks.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "PiiRedactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies configured PII redaction patterns to a single string.

        Args:
            text: The string to be redacted.

        Returns:
            The string with identified PII redacted.
        """
        if not isinstance(text, str):
            logger.warning(
                "Expected string for _redact_string, but received type '%s'. Returning data unchanged.",
                type(text)
            )
            return text

        redacted_text = text
        for pii_type, (pattern, replacement) in self._REDACTION_PATTERNS.items():
            try:
                redacted_text = pattern.sub(replacement, redacted_text)
                logger.debug("Successfully applied '%s' redaction pattern.", pii_type)
            except Exception as e:
                # Catching generic Exception for robustness against unforeseen regex issues
                logger.error(
                    "Failed to apply '%s' redaction pattern '%s': %s",
                    pii_type, pattern.pattern, e
                )
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        The method handles different data types:
        - If `data` is a string, it directly redacts PII within that string.
        - If `data` is a dictionary, it recursively processes string values and
          nested dictionaries/lists within it.
        - If `data` is a list, it recursively processes each item in the list.
        - For other data types, it returns the data unchanged, logging a warning.

        Args:
            data: The input data, which can be a string, dict, or list.
            context: A dictionary containing contextual information for the node's operation.
                     Not directly used for redaction logic in this version but available.

        Returns:
            The processed data with PII redacted.
        """
        logger.info("PiiRedactorNode received data for processing. Context: %s", context)

        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            redacted_data = {}
            for key, value in data.items():
                redacted_data[key] = self.process(value, context) # Recursive call for dict values
            return redacted_data
        elif isinstance(data, list):
            redacted_list = []
            for item in data:
                redacted_list.append(self.process(item, context)) # Recursive call for list items
            return redacted_list
        else:
            logger.warning(
                "PiiRedactorNode received unsupported data type '%s'. Returning data unchanged.",
                type(data)
            )
            return data