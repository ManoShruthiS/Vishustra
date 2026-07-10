import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node designed to redact common Personally Identifiable Information (PII)
    from text data.

    This node employs regular expressions to identify and replace sensitive data patterns
    such as email addresses, phone numbers, Social Security Numbers (SSN), and
    credit card numbers with generic redaction placeholders.
    """

    # Compiled regular expressions for various PII types and their corresponding
    # redaction placeholders. These are defined at the class level for efficiency.
    _PII_PATTERNS = {
        "email": (re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}\b'), '[EMAIL_REDACTED]'),
        "phone": (re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[PHONE_REDACTED]'),
        "ssn": (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN_REDACTED]'),
        "credit_card": (re.compile(r'\b(?:\d{4}[ -]?){3}\d{4}\b'), '[CREDIT_CARD_REDACTED]'), # Covers common 16-digit formats
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "PII Redactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, identifying and redacting PII patterns.

        The `process` method iterates through predefined PII patterns. If the
        input `data` is a string, it replaces all occurrences of these patterns
        with generic redaction placeholders. Non-string data will result in a
        TypeError.

        Args:
            data: The input data to be processed. Expected to be a string
                  containing text potentially with PII.
            context: A dictionary providing runtime context. This node does not
                     currently use context for its core logic, but it's available
                     for future extensions (e.g., dynamic pattern loading).

        Returns:
            The processed data with PII redacted. If the input `data` was a string,
            a new string with redactions will be returned.

        Raises:
            TypeError: If the input `data` is not a string, as this node
                       is designed specifically for text processing.
        """
        if not isinstance(data, str):
            logger.error(
                "PIIRedactorNode received non-string data. "
                f"Expected string, but got type: {type(data)}. Cannot redact PII."
            )
            raise TypeError(
                f"PIIRedactorNode expects string data for redaction, "
                f"but received {type(data)}."
            )

        if not data:
            logger.debug("PIIRedactorNode received an empty string. No redaction needed.")
            return data

        redacted_data = data
        for pii_type, (pattern, replacement) in self._PII_PATTERNS.items():
            # Perform replacement and count occurrences
            new_redacted_data, num_replacements = pattern.subn(replacement, redacted_data)

            if num_replacements > 0:
                logger.info(f"Redacted {num_replacements} instance(s) of {pii_type}.")
                redacted_data = new_redacted_data
            else:
                logger.debug(f"No instances of {pii_type} found for redaction.")

        logger.debug("PII Redaction process completed.")
        return redacted_data