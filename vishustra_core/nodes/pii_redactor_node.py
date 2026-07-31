import logging
import re
from typing import Any, Dict, List, Union

# Assuming BaseNode is available via this path as per instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts personally identifiable information (PII)
    from text data.

    This node identifies common PII patterns (e.g., email addresses, phone numbers,
    SSNs, IP addresses, credit card numbers, URLs) within string data and replaces
    them with generic redaction placeholders (e.g., '[REDACTED_EMAIL]').
    It supports processing of strings, dictionaries, and lists, recursively
    traversing nested structures.
    """

    _PII_PATTERNS: Dict[str, str] = {
        # Email addresses
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}": "[REDACTED_EMAIL]",
        # Phone numbers: common formats like (123) 456-7890, 123-456-7890, +1 123 456 7890
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b": "[REDACTED_PHONE]",
        # Social Security Numbers (format XXX-XX-XXXX)
        r"\b\d{3}-\d{2}-\d{4}\b": "[REDACTED_SSN]",
        # IP Addresses (IPv4)
        r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b": "[REDACTED_IP_ADDRESS]",
        # Generic Credit Card Numbers (simplified, 16 digits with spaces/hyphens)
        r"\b(?:\d{4}[- ]){3}\d{4}\b": "[REDACTED_CREDIT_CARD]",
        # URLs (often contain sensitive info or just to clean up)
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+": "[REDACTED_URL]",
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies PII redaction patterns to a single string.
        """
        if not isinstance(text, str):
            logger.warning(
                f"[{self.node_name}] Expected string for redaction, but received {type(text)}. Skipping."
            )
            return text

        redacted_text = text
        for pattern, replacement in self._PII_PATTERNS.items():
            # Only update if a match is found to avoid unnecessary re-assignments
            if re.search(pattern, redacted_text):
                original_len = len(redacted_text)
                redacted_text = re.sub(pattern, replacement, redacted_text)
                if len(redacted_text) != original_len: # Check if actual change occurred
                    logger.debug(
                        f"[{self.node_name}] Redacted '{replacement}' using pattern '{pattern}' in data."
                    )
        return redacted_text

    def _traverse_and_redact(self, data: Any) -> Any:
        """
        Recursively traverses data structures (dict, list, string) and redacts PII.
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            # Recursively process values in a dictionary
            return {k: self._traverse_and_redact(v) for k, v in data.items()}
        elif isinstance(data, list):
            # Recursively process elements in a list
            return [self._traverse_and_redact(item) for item in data]
        else:
            # For other primitive types (int, float, bool, None), return as is.
            # Log a debug message if it's an unexpected complex type that's not being processed.
            if not isinstance(data, (int, float, bool, type(None))):
                logger.debug(
                    f"[{self.node_name}] Encountered non-string, non-dict, non-list data of type {type(data)}. Returning as is without redaction."
                )
            return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, redacting PII from strings within it.
        Supports string, dict, and list inputs, including nested structures.

        Args:
            data: The input data, which can be a string, a dictionary, or a list
                  containing strings or nested structures.
            context: A dictionary containing contextual information for the node's operation.
                     Currently not used by this node but available for future extensions.

        Returns:
            The data with identified PII redacted.

        Raises:
            Exception: If an unexpected error occurs during the redaction process.
        """
        logger.info(f"[{self.node_name}] Starting PII redaction process.")
        try:
            processed_data = self._traverse_and_redact(data)
            logger.info(f"[{self.node_name}] PII redaction process completed successfully.")
            return processed_data
        except Exception as e:
            logger.error(
                f"[{self.node_name}] Critical error during PII redaction: {e}", exc_info=True
            )
            # Re-raise the exception to indicate a processing failure in the orchestration
            raise ValueError(f"Failed to redact PII: {e}") from e
