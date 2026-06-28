import re
import logging
from typing import Any, Dict, List, Tuple

# Assuming the base_node is located as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node responsible for redacting Personally Identifiable
    Information (PII) from input data. It provides a configurable mechanism
    to identify and replace sensitive data patterns within strings, and
    recursively processes string values within dictionaries and lists.

    Current default redaction patterns target common PII types such as:
    - Email addresses
    - US Phone numbers (various formats)
    - US Social Security Numbers (SSN-like patterns)
    """

    # Pre-compiled regex patterns for efficiency and their corresponding replacements.
    # The order of patterns might matter for overlapping definitions.
    _REDACTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
        # Email addresses (e.g., user@example.com)
        (re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", re.IGNORECASE), "[EMAIL_REDACTED]"),
        # US Phone numbers (e.g., 123-456-7890, (123) 456-7890, 123.456.7890)
        (re.compile(r"\b(?:\d{3}[-.\s]?|\(\d{3}\)\s?)\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
        # US Social Security Number (e.g., XXX-XX-XXXX)
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
        # Generic phrases that might indicate PII, like "My name is John Doe"
        # For a production system, more sophisticated NLP or entity recognition
        # would be used for names, addresses, etc. This is a basic example.
        (re.compile(r"my name is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", re.IGNORECASE), "my name is [NAME_REDACTED]"),
    ]

    def __init__(self):
        """
        Initializes the PII Redactor Node.
        """
        logger.debug(f"[{self.node_name}] Initializing PII Redactor Node.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "PII Redactor Node"

    def _redact_string(self, text: str) -> str:
        """
        Applies all defined redaction patterns to a given string.

        Args:
            text: The input string to be redacted.

        Returns:
            The string with all identified PII patterns replaced by their
            respective redaction placeholders.
        """
        original_text = text
        for pattern, replacement in self._REDACTION_PATTERNS:
            text = pattern.sub(replacement, text)

        if original_text != text:
            logger.debug(f"[{self.node_name}] Redaction applied to a string. Original length: {len(original_text)}, New length: {len(text)}")
        return text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        This method supports:
        - Direct string input: Redacts PII within the string.
        - Dictionary input: Recursively redacts PII within string values
          and within nested dictionaries/lists.
        - List input: Iterates through the list, recursively processing
          string or dictionary items.

        Args:
            data: The input data payload. Expected to be a string,
                  dictionary, or a list containing strings/dictionaries.
            context: A dictionary containing contextual information
                     relevant to the current processing pipeline.
                     While not directly used for configuration in this
                     basic implementation, it can be extended to pass
                     custom redaction rules or flags in advanced scenarios.

        Returns:
            The processed data with PII redacted. The structure of the
            data is preserved.

        Raises:
            TypeError: If the input data type is not supported for redaction.
        """
        logger.info(f"[{self.node_name}] Starting PII redaction process for input data type: {type(data)}.")

        if isinstance(data, str):
            redacted_data = self._redact_string(data)
            logger.debug(f"[{self.node_name}] Processed string data.")
            return redacted_data
        elif isinstance(data, dict):
            redacted_dict = {}
            for key, value in data.items():
                # Recursively process values that could contain PII
                if isinstance(value, (str, dict, list)):
                    logger.debug(f"[{self.node_name}] Recursing into dictionary key '{key}'.")
                    redacted_dict[key] = self.process(value, context)
                else:
                    redacted_dict[key] = value # Preserve non-string/dict/list values
            logger.debug(f"[{self.node_name}] Processed dictionary data.")
            return redacted_dict
        elif isinstance(data, list):
            redacted_list = []
            for index, item in enumerate(data):
                # Recursively process list items that could contain PII
                if isinstance(item, (str, dict, list)):
                    logger.debug(f"[{self.node_name}] Recursing into list item at index {index}.")
                    redacted_list.append(self.process(item, context))
                else:
                    redacted_list.append(item) # Preserve non-string/dict/list items
            logger.debug(f"[{self.node_name}] Processed list data.")
            return redacted_list
        else:
            error_msg = (
                f"[{self.node_name}] Unsupported data type for PII redaction: {type(data)}. "
                "Expected str, dict, or list of str/dict."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)