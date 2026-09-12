import logging
import re
from typing import Any, Dict, Union

# Assuming this path exists within the Vishustra framework structure.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node designed to redact Personally Identifiable Information (PII)
    from input data. It employs regular expressions to identify common PII patterns
    such as email addresses, phone numbers, social security number (SSN)-like patterns,
    IP addresses, and credit card numbers, replacing them with generic
    '[REDACTED_TYPE]' placeholders.

    This node is capable of performing redaction within string inputs, or within
    string values embedded in dictionaries and lists, providing a flexible
    mechanism for PII masking across various data structures.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor"

    def _redact_text(self, text: str) -> str:
        """
        Applies a series of regex patterns to redact PII from a given string.

        Args:
            text (str): The input string to be redacted.

        Returns:
            str: The string with identified PII replaced by redaction placeholders.
        """
        if not isinstance(text, str):
            logger.debug(f"Input for _redact_text was not a string ({type(text)}), returning as-is.")
            return text

        redacted_text = text

        # Email addresses (e.g., user@example.com)
        redacted_text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]', redacted_text)

        # Phone numbers (various common formats: (123) 456-7890, 123-456-7890, 123.456.7890, +1 123-456-7890)
        # This pattern is robust but might have false positives if a sequence of numbers happens to match.
        redacted_text = re.sub(
            r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)',
            '[REDACTED_PHONE]',
            redacted_text
        )

        # Social Security Number (SSN)-like pattern (e.g., XXX-XX-XXXX, XXX XX XXXX, XXX.XX.XXXX)
        redacted_text = re.sub(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', '[REDACTED_SSN]', redacted_text)

        # IP Addresses (e.g., 192.168.1.1)
        redacted_text = re.sub(
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            '[REDACTED_IP]',
            redacted_text
        )

        # Credit Card Numbers (simple 13-16 digit sequence, with optional spaces/hyphens)
        # WARNING: For production environments, use dedicated PCI-compliant tokenization services
        # or libraries for secure credit card handling, as simple regex is insufficient.
        redacted_text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[REDACTED_CREDIT_CARD]', redacted_text)

        # Note: Redacting named entities (like personal names) accurately requires
        # Named Entity Recognition (NER) models (e.g., spaCy, NLTK) rather than
        # simple regex, to avoid over-redaction or misses. This node focuses on
        # pattern-based PII.

        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact common PII patterns.
        This method supports recursive redaction for nested data structures
        like lists and dictionaries.

        Args:
            data (Any): The input data. This can be a string, a list of strings
                        or nested structures, or a dictionary where values can
                        be strings or nested structures.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing pipeline. Currently, this
                                      node does not utilize context for its redaction
                                      rules but it is available for future enhancements.

        Returns:
            Any: The processed data with identified PII redacted. If the input data type
                 is not directly supported for string-based redaction (e.g., an integer
                 or boolean), a warning is logged, and the original data is returned
                 unmodified to prevent pipeline breakage.
        """
        if isinstance(data, str):
            logger.debug("Attempting PII redaction on a string input.")
            return self._redact_text(data)
        elif isinstance(data, list):
            logger.debug(f"Recursively redacting PII from a list with {len(data)} items.")
            # Recursively call process for each item in the list
            return [self.process(item, context) for item in data]
        elif isinstance(data, dict):
            logger.debug(f"Recursively redacting PII from a dictionary with {len(data)} keys.")
            # Recursively call process for each value in the dictionary
            return {key: self.process(value, context) for key, value in data.items()}
        elif data is None:
            logger.debug("Input data is None, returning as-is.")
            return None
        else:
            # For types that cannot be processed for string PII (e.g., int, float, bool),
            # log a warning and return the data unmodified.
            logger.warning(
                f"Unsupported data type for direct PII redaction: {type(data).__name__}. "
                "Returning data without modification. If PII is expected in this format, "
                "consider pre-processing to convert to string or supported structures."
            )
            return data