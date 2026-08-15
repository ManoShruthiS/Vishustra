import re
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node that redacts common Personally Identifiable Information (PII)
    from text data using regular expressions.

    This node is designed to identify and mask sensitive patterns such as email addresses,
    phone numbers, IP addresses, and common ID formats (e.g., SSN-like patterns)
    within string inputs.
    """

    def __init__(self, redaction_string: str = "[REDACTED_PII]"):
        """
        Initializes the PIIRedactorNode with a configurable redaction string.

        Args:
            redaction_string: The string to replace identified PII with.
                              Defaults to "[REDACTED_PII]".
        """
        self._redaction_string = redaction_string
        # Define common PII patterns. These are illustrative and cover frequently
        # encountered formats. For highly robust, context-aware PII detection,
        # integration with specialized NLP libraries is recommended.
        self._pii_patterns = {
            "email_address": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            "phone_number": re.compile(
                r'\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
                # Covers formats like (123) 456-7890, 123-456-7890, +1 123 456 7890
            ),
            "ip_address": re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ),
            # Simple simulation for a US SSN-like pattern. Not intended for global ID
            # detection or secure validation.
            "ssn_like": re.compile(
                r'\b\d{3}-\d{2}-\d{4}\b'
            ),
        }
        logger.debug(f"PIIRedactorNode initialized with redaction string: '{self._redaction_string}'")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PIIRedactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact common PII patterns.

        The `data` is expected to be a string. If `data` is not a string, a
        `TypeError` will be raised.

        Args:
            data: The input data, which must be a string containing text to be
                  scanned for PII.
            context: A dictionary containing contextual information. This parameter
                     is currently not utilized by the PIIRedactorNode but is
                     available for future extensions (e.g., to configure PII
                     types to redact dynamically).

        Returns:
            The processed string with all identified PII patterns replaced by the
            configured redaction string.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"PIIRedactorNode expects string data, but received "
                f"'{type(data).__name__}'. Unable to perform PII redaction."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        redacted_data = str(data)  # Create a mutable copy of the input string
        
        logger.info(f"Starting PII redaction process. Input data (first 100 chars): "
                    f"'{redacted_data[:100]}{'...' if len(redacted_data) > 100 else ''}'")

        for pii_type, pattern in self._pii_patterns.items():
            original_match_count = len(pattern.findall(redacted_data))
            if original_match_count > 0:
                redacted_data = pattern.sub(self._redaction_string, redacted_data)
                logger.debug(f"Redacted {original_match_count} instance(s) of '{pii_type}' PII.")
            else:
                logger.debug(f"No '{pii_type}' PII found in the current data segment.")

        logger.info("PII redaction complete for this data segment.")
        return redacted_data