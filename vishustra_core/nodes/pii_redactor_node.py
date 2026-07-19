import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node that redacts personally identifiable information (PII)
    from input data. It scans for common PII patterns like email addresses,
    phone numbers, and social security numbers (US format) within strings
    and replaces them with generic placeholders.

    Supports redaction within strings, lists, and dictionaries, recursively
    traversing complex data structures.
    """

    def __init__(self):
        """
        Initializes the PII Redactor Node with a predefined set of
        PII detection patterns. Each pattern includes a name, the regex pattern,
        and the placeholder to use for redaction.
        """
        self._pii_patterns: List[Dict[str, str]] = [
            {
                "name": "email",
                "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "placeholder": "[REDACTED_EMAIL]"
            },
            {
                "name": "us_phone_number",
                # Matches various US phone number formats: (123) 456-7890, 123-456-7890, 123.456.7890, 1234567890
                # Optionally includes country code like +1
                "pattern": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                "placeholder": "[REDACTED_PHONE]"
            },
            {
                "name": "us_ssn",
                "pattern": r'\b\d{3}-\d{2}-\d{4}\b',
                "placeholder": "[REDACTED_SSN]"
            },
            {
                "name": "ip_address",
                # Matches IPv4 addresses
                "pattern": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                "placeholder": "[REDACTED_IP]"
            }
        ]
        logger.debug(f"{self.node_name} initialized with {len(self._pii_patterns)} PII patterns.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PIIRedactorNode"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured redaction patterns to a given string.
        """
        redacted_text = text
        for pii_type in self._pii_patterns:
            try:
                redacted_text = re.sub(pii_type["pattern"], pii_type["placeholder"], redacted_text)
            except re.error as e:
                logger.error(f"[{self.node_name}] Failed to apply regex pattern '{pii_type['pattern']}' for {pii_type['name']}: {e}")
                # Continue with other patterns even if one fails
            except Exception as e:
                logger.error(f"[{self.node_name}] An unexpected error occurred while redacting {pii_type['name']}: {e}")
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        If the data is a string, it applies PII redaction directly.
        If the data is a list or dictionary, it recursively applies redaction
        to all string elements or values within them.
        For other data types, the data is returned unchanged.

        Args:
            data (Any): The input data to be processed. Can be a string,
                        list, dictionary, or any other type.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     for the processing operation. While not directly
                                     used for PII patterns in this implementation,
                                     it's available for future extensions (e.g., passing
                                     dynamic patterns or redaction preferences).

        Returns:
            Any: The data with PII redacted. If the input data type is not
                 a string, list, or dictionary, it is returned as-is.
        """
        if not isinstance(context, dict):
            logger.warning(f"[{self.node_name}] Provided context is not a dictionary. Proceeding with an empty context.")
            context = {} # Ensure context is a dict for safety

        logger.debug(f"[{self.node_name}] Received data of type: {type(data)}")

        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, list):
            # Recursively process each item in the list
            return [self.process(item, context) for item in data]
        elif isinstance(data, dict):
            # Recursively process each value in the dictionary
            return {key: self.process(value, context) for key, value in data.items()}
        else:
            # For other types (int, float, bool, None, custom objects), return as-is
            logger.debug(f"[{self.node_name}] Data type {type(data)} not suitable for PII redaction. Returning as-is.")
            return data