import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node designed to redact common Personally Identifiable Information (PII)
    from textual data.

    This node utilizes regular expressions to identify and replace patterns corresponding to
    emails, common phone numbers, and Social Security Numbers (SSNs) with generic,
    redacted placeholders (e.g., [REDACTED_EMAIL]). It primarily operates on string input.
    Non-string data is logged with a warning and passed through without modification.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "PII Redactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII patterns.

        If the input `data` is a string, it applies a series of regular expression-based
        redactions for known PII types. If `data` is not a string, it logs a warning
        and returns the data as-is, as redaction logic is string-specific.

        Args:
            data: The input data, ideally a string expected to contain text that
                  may need PII redaction.
            context: A dictionary containing contextual information for processing.
                     This node currently does not utilize the context, but it's
                     part of the `BaseNode` interface definition.

        Returns:
            The processed data with identified PII redacted, or the original data
            if it was not a string or no PII was found.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data (type: {type(data).__name__}). "
                "PII redaction requires string input; data passed through untouched."
            )
            return data

        # Ensure we are working with a mutable string copy
        redacted_data = str(data)
        original_data = str(data) # Keep a copy to check if any changes occurred

        # --- PII Redaction Patterns ---
        # These patterns are illustrative and cover common formats.
        # Production systems might require more comprehensive and internationalized patterns.

        # 1. Email addresses: e.g., name@domain.com
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, redacted_data):
            redacted_data = re.sub(email_pattern, '[REDACTED_EMAIL]', redacted_data)
            logger.debug(f"[{self.node_name}] Detected and redacted email addresses.")

        # 2. US Phone numbers: e.g., (123) 456-7890, 123-456-7890, 123.456.7890, 123 456 7890
        phone_pattern = r'\b(?:\d{3}[-.\s]?|\(\d{3}\)\s?)\d{3}[-.\s]?\d{4}\b'
        if re.search(phone_pattern, redacted_data):
            redacted_data = re.sub(phone_pattern, '[REDACTED_PHONE]', redacted_data)
            logger.debug(f"[{self.node_name}] Detected and redacted phone numbers.")

        # 3. US Social Security Numbers (SSN): e.g., XXX-XX-XXXX
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        if re.search(ssn_pattern, redacted_data):
            redacted_data = re.sub(ssn_pattern, '[REDACTED_SSN]', redacted_data)
            logger.debug(f"[{self.node_name}] Detected and redacted SSN patterns.")

        # --- Logging Result ---
        if redacted_data != original_data:
            logger.info(f"[{self.node_name}] Successfully redacted PII from the input data.")
        else:
            logger.debug(f"[{self.node_name}] No PII patterns found or redacted in the data.")

        return redacted_data