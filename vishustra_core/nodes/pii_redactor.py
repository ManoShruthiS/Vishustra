import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra node designed to redact personally identifiable information (PII)
    from text data. It applies a set of predefined regex patterns to identify
    and replace sensitive information with generic placeholders.

    This node supports redaction within strings, and recursively processes
    string values found within dictionaries and lists, preserving the overall
    data structure.
    """

    # Predefined PII patterns and their corresponding redaction placeholders.
    # These patterns are illustrative and can be extended or made configurable
    # for production deployments.
    _PII_PATTERNS = {
        # Email addresses (e.g., user@example.com)
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'): '[EMAIL_REDACTED]',
        # Common US phone number formats (e.g., 123-456-7890, (123) 456-7890)
        re.compile(r'\b(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'): '[PHONE_REDACTED]',
        # US Social Security Numbers (e.g., 123-45-6789)
        re.compile(r'\b\d{3}-\d{2}-\d{4}\b'): '[SSN_REDACTED]',
        # Credit Card Numbers (basic 13-16 digit patterns, without Luhn validation)
        re.compile(r'\b(?:\d[ -]*?){13,16}\b'): '[CREDIT_CARD_REDACTED]',
        # IP Addresses (IPv4, e.g., 192.168.1.1)
        re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'): '[IP_REDACTED]',
    }

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies all defined PII redaction patterns to a given string.

        Args:
            text: The input string potentially containing PII.

        Returns:
            The string with identified PII replaced by placeholders.
        """
        redacted_text = text
        for pattern, placeholder in self._PII_PATTERNS.items():
            redacted_text = pattern.sub(placeholder, redacted_text)
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        This method supports:
        -   `str`: Direct redaction using defined patterns.
        -   `dict`: Recursively processes string values within the dictionary.
        -   `list`: Recursively processes string elements within the list.

        For other data types, the data is returned unchanged, and a warning
        is logged.

        Args:
            data: The input data, which can be a string, dictionary, list,
                  or any other type.
            context: A dictionary containing contextual information for the node's
                     operation. While not used for PII rules in this base implementation,
                     it provides extensibility for dynamic rule application.

        Returns:
            The processed data with PII redacted, maintaining its original structure.
            If the data type is unsupported for redaction, the original data is returned.
        """
        try:
            if isinstance(data, str):
                logger.debug("Attempting PII redaction on string data.")
                return self._redact_string(data)
            elif isinstance(data, dict):
                logger.debug("Recursively processing dictionary for PII redaction.")
                redacted_dict = {}
                for key, value in data.items():
                    redacted_dict[key] = self.process(value, context)  # Recursive call
                return redacted_dict
            elif isinstance(data, list):
                logger.debug("Recursively processing list for PII redaction.")
                redacted_list = []
                for item in data:
                    redacted_list.append(self.process(item, context))  # Recursive call
                return redacted_list
            else:
                logger.info(
                    f"Skipping PII redaction for unsupported data type: {type(data).__name__}. "
                    "Data must be a string, dict, or list."
                )
                return data
        except Exception as e:
            logger.error(f"An error occurred during PII redaction: {e}", exc_info=True)
            # In case of an unexpected error, return the original data to prevent data loss.
            return data