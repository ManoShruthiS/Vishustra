import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts personally identifiable information (PII)
    from text data. This node identifies common PII patterns like email addresses
    and phone numbers and replaces them with a generic '[REDACTED]' placeholder.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PIIRedactor"

    def __init__(self):
        """
        Initializes the PIIRedactorNode with predefined PII patterns.
        Regex patterns are compiled for efficiency.
        """
        # Define regex patterns for common PII types
        self._pii_patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
            # Future expansion: Add more patterns for other PII types (e.g., SSN, credit card numbers)
        }
        self._redaction_placeholders = {
            "email": "[EMAIL_REDACTED]",
            "phone": "[PHONE_REDACTED]"
            # Corresponding placeholders for other PII types
        }
        logger.debug(f"{self.node_name} initialized with patterns for: {list(self._pii_patterns.keys())}")


    def _redact_string(self, text: str) -> str:
        """
        Applies PII redaction to a single string using configured patterns.

        Args:
            text: The input string to redact.

        Returns:
            The string with PII replaced by placeholders.
        """
        redacted_text = text
        for pii_type, pattern in self._pii_patterns.items():
            placeholder = self._redaction_placeholders.get(pii_type, "[REDACTED]")
            
            # Use finditer for potentially better performance/memory for many matches,
            # but sub is fine for direct replacement and simpler.
            matches_found = 0
            # To ensure the pattern replacement happens iteratively and correctly
            # without re-matching a placeholder if it contains sub-strings of PII.
            # For simple patterns like email/phone, direct sub is usually safe.
            if pattern.search(redacted_text): # Quick check if any match exists
                original_matches = pattern.findall(redacted_text)
                matches_found = len(original_matches)
                redacted_text = pattern.sub(placeholder, redacted_text)
            
            if matches_found > 0:
                logger.debug(
                    f"Redacted {matches_found} instances of '{pii_type}' "
                    f"with '{placeholder}' in the string."
                )
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        This method supports:
        -   `str`: Redacts PII directly from the string.
        -   `list[str]`: Processes each string item in the list. Non-string items are kept as-is.
        -   `dict`: Processes string values and list of string values within the dictionary.
                    Other value types are kept as-is.

        Unsupported data types will be logged and returned without modification.

        Args:
            data: The input data to be processed for PII redaction.
                  Expected types are str, list[str], or dict with string/list[str] values.
            context: A dictionary containing contextual information for processing.
                     (Not directly used for redaction logic in this version, but available
                     for potential future configuration, e.g., enabling/disabling patterns).

        Returns:
            The data with PII redacted, or the original data if its type is unsupported
            or an unrecoverable error occurs during redaction.
        """
        if data is None:
            logger.debug(f"{self.node_name} received None data. Returning as is.")
            return None

        try:
            if isinstance(data, str):
                logger.debug(f"{self.node_name} processing a single string.")
                return self._redact_string(data)
            elif isinstance(data, list):
                logger.debug(f"{self.node_name} processing a list of items.")
                # Process each string in the list; keep non-string items as they are
                return [self._redact_string(item) if isinstance(item, str) else item for item in data]
            elif isinstance(data, dict):
                logger.debug(f"{self.node_name} processing a dictionary.")
                redacted_dict = {}
                for key, value in data.items():
                    if isinstance(value, str):
                        redacted_dict[key] = self._redact_string(value)
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        # If it's a list of strings, process each
                        redacted_dict[key] = [self._redact_string(item) for item in value]
                    else:
                        # For other types of values (e.g., int, bool, nested dicts),
                        # keep them as they are without attempting redaction.
                        redacted_dict[key] = value
                return redacted_dict
            else:
                logger.warning(
                    f"{self.node_name} received unsupported data type: {type(data).__name__}. "
                    "Data will be returned without PII redaction."
                )
                return data
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during PII redaction in {self.node_name}: {e}",
                exc_info=True  # Log full traceback for debugging
            )
            # In case of an error, it's generally safer to return the original data
            # to prevent accidental data loss or corruption further down the pipeline.
            return data