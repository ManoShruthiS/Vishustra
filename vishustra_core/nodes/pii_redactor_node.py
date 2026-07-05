
import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node for Vishustra that redacts personally identifiable information (PII)
    from text data using predefined regular expression patterns.

    This node is designed to identify common PII types such as email addresses,
    phone numbers, and basic patterns resembling social security numbers,
    replacing them with generic redacted placeholders.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PIIRedactor"

    def __init__(self):
        """
        Initializes the PIIRedactorNode with a set of regular expressions
        for common PII patterns and their corresponding redaction placeholders.
        """
        # Define PII patterns and their replacements.
        # Note: Robust PII detection, especially for names or complex identifiers,
        # often requires more sophisticated NLP techniques (e.g., Named Entity Recognition).
        # This implementation focuses on regex-identifiable patterns.
        self._pii_patterns = {
            "email": {
                "regex": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "replacement": "[REDACTED_EMAIL]"
            },
            "phone_number": {
                "regex": r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                "replacement": "[REDACTED_PHONE]"
            },
            # A simple pattern resembling a US Social Security Number format (e.g., XXX-XX-XXXX).
            # This is illustrative; real-world SSN detection must be more robust and context-aware.
            "ssn_like": {
                "regex": r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
                "replacement": "[REDACTED_SSN]"
            }
        }
        logger.debug(f"{self.node_name} node initialized with {len(self._pii_patterns)} PII patterns loaded.")

    def _redact_text(self, text: str) -> str:
        """
        Applies all configured PII redaction patterns to a given string.

        Args:
            text: The input string to be redacted.

        Returns:
            The string with identified PII replaced by placeholders.
        """
        redacted_text = text
        for pii_type, config in self._pii_patterns.items():
            try:
                pattern = re.compile(config["regex"])
                replacement = config["replacement"]
                # Only perform substitution if the pattern is found to avoid unnecessary work and logging noise.
                if pattern.search(redacted_text):
                    logger.debug(f"Detected and redacting '{pii_type}' in text.")
                    redacted_text = pattern.sub(replacement, redacted_text)
            except re.error as e:
                logger.error(f"Regex error for pattern '{pii_type}': {e}. Skipping this pattern.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during redaction of '{pii_type}': {e}.")
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        This method supports redaction for strings, lists of strings,
        and dictionaries where values might be strings or nested structures.
        Non-string elements in collections, or standalone non-string data,
        are passed through unchanged.

        Args:
            data: The input data, which can be a `str`, `list`, `dict`, or any other type.
            context: A dictionary containing context-specific information for the current
                     orchestration run. This node does not currently utilize the context,
                     but it's available for future configuration or dynamic behavior.

        Returns:
            The input data with PII redacted. If the input data type is not supported
            for redaction (e.g., an integer), it is returned as-is.
        """
        logger.info(f"{self.node_name} initiated PII redaction for incoming data.")

        if isinstance(data, str):
            # Process a single string
            return self._redact_text(data)
        elif isinstance(data, list):
            # Recursively process each item in a list
            redacted_list = [self.process(item, context) for item in data]
            return redacted_list
        elif isinstance(data, dict):
            # Recursively process each value in a dictionary
            redacted_dict = {key: self.process(value, context) for key, value in data.items()}
            return redacted_dict
        else:
            # For any other data type, return it as-is without modification.
            # Log a warning if it's not None, as None might be an expected "no data" case.
            if data is not None:
                logger.warning(
                    f"{self.node_name} received unsupported data type '{type(data).__name__}' "
                    "for redaction. Returning data without modification."
                )
            return data

