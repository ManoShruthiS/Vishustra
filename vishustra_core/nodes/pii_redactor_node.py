import logging
import re
from typing import Any, Dict, Union, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node responsible for redacting Personally Identifiable
    Information (PII) from input data.

    This node traverses common data structures (strings, dictionaries, lists)
    and applies regular expression patterns to identify and replace PII
    with a generic redaction string.
    """

    # Predefined regular expression patterns for common PII types.
    # These are illustrative and can be expanded or made configurable.
    _PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', # e.g., +1 (123) 456-7890, 123-456-7890
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b', # US SSN format
        "credit_card": r'\b(?:\d[ -]*?){13,16}\b' # Matches 13 to 16 digits, with optional spaces/hyphens
    }
    _REDACTION_STRING = "[REDACTED_PII]"

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured PII redaction patterns to a given string.
        """
        redacted_text = text
        for pii_type, pattern in self._PII_PATTERNS.items():
            try:
                redacted_text = re.sub(pattern, self._REDACTION_STRING, redacted_text)
            except re.error as e:
                logger.warning(
                    f"PII Redactor: Failed to apply regex pattern for '{pii_type}'. "
                    f"Pattern might be malformed: {e}"
                )
            except Exception as e:
                logger.error(f"PII Redactor: An unexpected error occurred during string redaction for '{pii_type}': {e}")
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        This method recursively traverses `str`, `dict`, and `list` types.
        Strings are subjected to regex-based PII detection and redaction.
        Other data types are returned as they are, with a debug log entry.

        Args:
            data: The input data to be processed. Can be a string, dict, list,
                  or any other Python type.
            context: A dictionary providing additional runtime context or configuration.
                     This implementation does not currently use the context for
                     configuration but passes it recursively.

        Returns:
            The processed data with identified PII redacted, or the original
            data if its type is not handled by the redaction logic.
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            redacted_data = {}
            for key, value in data.items():
                # Recursively process values in dictionaries
                redacted_data[key] = self.process(value, context)
            return redacted_data
        elif isinstance(data, list):
            redacted_data = []
            for item in data:
                # Recursively process items in lists
                redacted_data.append(self.process(item, context))
            return redacted_data
        else:
            if data is not None:
                logger.debug(
                    f"PII Redactor: Received data of unhandled type '{type(data).__name__}'. "
                    "Returning data as-is without redaction."
                )
            return data