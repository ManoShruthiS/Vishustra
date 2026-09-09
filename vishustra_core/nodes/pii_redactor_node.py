
import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node responsible for redacting common Personally
    Identifiable Information (PII) patterns from text data.

    This node uses regular expressions to identify and replace sensitive data
    like email addresses, phone numbers, social security numbers, and credit
    card numbers with generic redaction placeholders.
    """

    # Class-level constants defining PII patterns and their respective
    # redaction replacements.
    # Each tuple contains (compiled_regex_pattern, replacement_string).
    _PII_PATTERNS = [
        # Email addresses: basic pattern for common email formats.
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[REDACTED_EMAIL]'),
        
        # Phone numbers: US-centric formats (XXX) XXX-XXXX, XXX-XXX-XXXX, XXXXXXXXXX.
        (re.compile(r'\b(?:\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4}|\d{10})\b'), '[REDACTED_PHONE]'),
        
        # Social Security Numbers (US format: XXX-XX-XXXX).
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[REDACTED_SSN]'),
        
        # Credit Card Numbers: Simplified pattern for 13-16 digits with optional
        # spaces or hyphens. Note: This is a heuristic and not a robust validator.
        (re.compile(r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[- ]?(?:\d{4}[- ]?){2}\d{4}\b'), '[REDACTED_CREDIT_CARD]'),
        
        # IP Addresses (IPv4 format).
        (re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'), '[REDACTED_IP_ADDRESS]'),
    ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact common PII.

        The method expects the input `data` to be a string. It iterates through
        predefined PII patterns, replacing any matches with a generic placeholder.

        Args:
            data: The input data to be processed. Expected to be a string
                  containing text.
            context: A dictionary containing contextual information for the
                     processing pipeline. While not directly used for pattern
                     configuration in this implementation, it's available for
                     future dynamic pattern loading or configuration.

        Returns:
            The input data with all identified PII patterns replaced by
            redaction placeholders.

        Raises:
            TypeError: If the input 'data' is not a string, indicating an
                       unsupported input type for text redaction.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected 'str', but received '%s'.",
                self.node_name,
                type(data).__name__
            )
            raise TypeError(
                f"[{self.node_name}] PII Redactor node requires string input for "
                f"redaction, but received {type(data).__name__}."
            )

        redacted_data = data
        total_redaction_count = 0

        for pattern, replacement in self._PII_PATTERNS:
            # Find all occurrences before performing replacement to get an accurate count
            matches = pattern.findall(redacted_data)
            if matches:
                num_matches = len(matches)
                redacted_data = pattern.sub(replacement, redacted_data)
                total_redaction_count += num_matches
                logger.debug(
                    "[%s] Redacted %d instance(s) of pattern '%s' (replaced with '%s').",
                    self.node_name,
                    num_matches,
                    pattern.pattern,
                    replacement
                )

        if total_redaction_count > 0:
            logger.info(
                "[%s] Successfully redacted %d PII instance(s) from the data.",
                self.node_name,
                total_redaction_count
            )
        else:
            logger.debug("[%s] No known PII instances found for redaction in the data.", self.node_name)

        return redacted_data

