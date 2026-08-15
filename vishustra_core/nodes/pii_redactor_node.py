import logging
import re
from typing import Any, Dict, Union, List, Tuple

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PiiRedactorNode(BaseNode):
    """
    A Vishustra node designed to redact personally identifiable information (PII)
    from text data. It uses regular expressions to identify and replace common
    PII patterns such as email addresses, phone numbers, and Social Security Numbers
    with configurable placeholders.

    This node supports recursive processing of string values within dictionaries and lists,
    ensuring comprehensive PII redaction across structured data payloads.
    """

    # Basic regex patterns for common PII types.
    # These patterns are designed to be reasonably robust but can be extended
    # or refined based on specific compliance requirements and data formats.
    _PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', # US-like phone numbers
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b', # US Social Security Numbers (format XXX-XX-XXXX)
        # Future enhancements could include patterns for credit card numbers,
        # IP addresses, physical addresses, or custom entity types.
    }

    def __init__(self,
                 redact_email: bool = True,
                 redact_phone: bool = True,
                 redact_ssn: bool = True,
                 redaction_placeholder: str = "[REDACTED]"):
        """
        Initializes the PII Redactor node with specific redaction policies.

        Args:
            redact_email (bool): If True, email addresses will be redacted.
            redact_phone (bool): If True, phone numbers will be redacted.
            redact_ssn (bool): If True, Social Security Numbers will be redacted.
            redaction_placeholder (str): The base string used as a placeholder
                                         for redacted PII. It will be augmented
                                         with the specific PII type (e.g., [REDACTED_EMAIL]).
        """
        self._redact_config = {
            "email": redact_email,
            "phone": redact_phone,
            "ssn": redact_ssn,
        }
        # Ensure placeholder is consistently formatted for augmented use.
        self._base_redaction_placeholder = redaction_placeholder.strip('[]')
        logger.info(f"PiiRedactorNode initialized. Redaction enabled for: "
                    f"{', '.join(k for k, v in self._redact_config.items() if v) or 'None'}.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "PiiRedactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies PII redaction to a single string based on the node's configured
        patterns and enabled types.

        Args:
            text (str): The input string to redact.

        Returns:
            str: The string with identified PII replaced by placeholders.
        """
        redacted_text = text
        for pii_type, enabled in self._redact_config.items():
            if enabled:
                pattern = self._PII_PATTERNS.get(pii_type)
                if pattern:
                    # Create a specific placeholder for the detected PII type
                    placeholder = f"[{self._base_redaction_placeholder}_{pii_type.upper()}]"
                    redacted_text = re.sub(pattern, placeholder, redacted_text)
                    logger.debug(f"Redacted PII of type '{pii_type}' in text segment.")
                else:
                    logger.warning(f"No regex pattern defined for PII type: '{pii_type}'. Skipping redaction for this type.")
        return redacted_text

    def _redact_recursive(self, data: Any) -> Any:
        """
        Recursively traverses data structures (dictionaries and lists/tuples)
        to apply PII redaction to all string values found within. Non-string
        primitive types are returned as-is.

        Args:
            data (Any): The data structure or primitive value to process.

        Returns:
            Any: The processed data structure with PII redacted from strings.
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            # Recursively process values in a dictionary
            return {k: self._redact_recursive(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            # Recursively process items in a list or tuple
            processed_items = [self._redact_recursive(item) for item in data]
            # Maintain the original type for lists and tuples
            return tuple(processed_items) if isinstance(data, tuple) else processed_items
        else:
            # Return any other data type (e.g., int, float, bool, None) as is.
            return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII according to
        the node's configuration. It handles various data types, recursively
        applying redaction to strings embedded within complex structures.

        Args:
            data (Any): The input data payload. This can be a string, a dictionary,
                        a list, or any other Python type. String values within
                        collections will be inspected and redacted.
            context (Dict[str, Any]): A dictionary providing contextual information
                                       for the current processing flow. While important
                                       for other nodes, this PII Redactor node does not
                                       directly use the context for its redaction logic.

        Returns:
            Any: The processed data payload with identified PII redacted.
                 If the input `data` is `None`, `None` is returned.
                 In case of an unexpected error during redaction, the original
                 data is returned after logging the exception, preventing flow interruption.
        """
        if data is None:
            logger.debug("Received None input data. Returning None.")
            return None

        try:
            redacted_data = self._redact_recursive(data)
            logger.debug("PII redaction process completed successfully.")
            return redacted_data
        except Exception as e:
            logger.error(f"An unexpected error occurred during PII redaction: {e}", exc_info=True)
            # In a robust framework, deciding whether to re-raise, return original data,
            # or return a specific error structure depends on the orchestration strategy.
            # Returning the original data is often a safer default to avoid crashing
            # the entire pipeline for a redaction failure, allowing downstream nodes
            # to potentially handle unredacted data or specific error flags.
            logger.warning("Returning original data due to a PII redaction error.")
            return data