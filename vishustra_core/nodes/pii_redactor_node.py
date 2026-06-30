import re
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts personally identifiable information (PII)
    from string data.

    This node uses regular expressions to identify and replace common PII patterns
    such as email addresses, phone numbers, and social security numbers with
    generic redacted placeholders.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII Redactor"

    def __init__(self):
        """
        Initializes the PIIRedactorNode with predefined PII redaction patterns.
        """
        self._pii_patterns = {
            # Email addresses
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'): '[EMAIL_REDACTED]',
            # US Phone numbers (various formats: (123) 456-7890, 123-456-7890, 123.456.7890, 123 456 7890)
            re.compile(r'(\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)|(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})'): '[PHONE_REDACTED]',
            # Social Security Numbers (simplified: DDD-DD-DDDD)
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'): '[SSN_REDACTED]',
            # Credit Card Numbers (simplified, 13-16 digits, with optional spaces/hyphens)
            re.compile(r'\b(?:\d[ -]*?){13,16}\b'): '[CREDIT_CARD_REDACTED]',
            # IP Addresses (IPv4 basic)
            re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'): '[IP_REDACTED]',
        }
        logger.debug("PIIRedactorNode initialized with default PII patterns.")

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        If the input `data` is a string, it applies predefined regular expressions
        to find and replace PII patterns. If `data` is not a string, it logs a
        warning and returns the data unchanged.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        containing text potentially with PII.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     for the processing node. Not directly used
                                     for redaction logic in this implementation,
                                     but available for future extensions (e.g.,
                                     custom patterns).

        Returns:
            Any: The processed data, with PII redacted if it was a string.
                 Returns the original data unchanged if it was not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"Node '{self.node_name}' received non-string data of type "
                f"'{type(data).__name__}'. Skipping PII redaction."
            )
            return data

        redacted_data = data
        for pattern, replacement in self._pii_patterns.items():
            initial_data = redacted_data
            redacted_data = pattern.sub(replacement, redacted_data)
            if initial_data != redacted_data:
                logger.debug(f"Pattern '{pattern.pattern}' applied, PII redacted.")

        if redacted_data != data:
            logger.info(f"Node '{self.node_name}' successfully redacted PII from data.")
        else:
            logger.debug(f"Node '{self.node_name}' processed data, no PII found or redacted.")

        return redacted_data
