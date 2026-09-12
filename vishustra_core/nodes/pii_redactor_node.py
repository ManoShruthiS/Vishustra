import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts Personally Identifiable Information (PII)
    from text data using regular expressions.

    This node is designed to identify and replace common PII patterns such as
    email addresses and phone numbers with a predefined placeholder. It can
    process individual strings, lists of strings, and dictionaries where values are strings.
    """

    # Regex patterns for common PII types
    # Email: Matches standard email formats (e.g., user@example.com)
    _EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    # Phone Number: Matches various international and national formats,
    # including those with spaces, hyphens, dots, or parentheses (e.g., +1 (123) 456-7890, 123-456-7890)
    _PHONE_PATTERN = re.compile(
        r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{2,5}\)|\d{2,5})[-.\s]?\d{2,5}[-.\s]?\d{2,6}\b'
    )
    # The above phone pattern is a bit more generic for international numbers.
    # For US-specific, one might use: r'\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b'
    # We will stick to the more generic one to broaden applicability.

    # Placeholder for redacted PII
    _REDACTION_PLACEHOLDER = "[REDACTED_PII]"

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies PII redaction patterns to a single string.
        """
        if not isinstance(text, str):
            logger.debug(
                f"[{self.node_name}] Attempted to redact non-string data internally. Returning as-is."
            )
            return text

        redacted_text = text
        # Redact email addresses
        redacted_text = self._EMAIL_PATTERN.sub(self._REDACTION_PLACEHOLDER, redacted_text)
        # Redact phone numbers
        redacted_text = self._PHONE_PATTERN.sub(self._REDACTION_PLACEHOLDER, redacted_text)
        
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        The `data` input can be:
        - A single string: PII patterns are applied directly.
        - A list of strings: Each string in the list is processed for PII.
          Non-string elements in the list are passed through unchanged.
        - A dictionary: Each string value in the dictionary is processed for PII.
          Keys are preserved, and non-string values are passed through unchanged.
        - Any other type: The data is returned unchanged, and a warning is logged,
          as this node is not designed to redact PII from non-textual data structures.

        Args:
            data (Any): The input data to be processed for PII redaction.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing pipeline. Not directly used
                                      by this node for its core redaction logic.

        Returns:
            Any: The data with identified PII redacted, or the original data if its type
                 is not supported for redaction.
        """
        logger.debug(f"[{self.node_name}] Starting PII redaction for data of type: {type(data)}")

        try:
            if isinstance(data, str):
                return self._redact_string(data)
            elif isinstance(data, list):
                # Process each item in the list, only redacting if it's a string
                return [
                    self._redact_string(item) if isinstance(item, str) else item
                    for item in data
                ]
            elif isinstance(data, dict):
                redacted_dict = {}
                # Process each string value in the dictionary
                for key, value in data.items():
                    if isinstance(value, str):
                        redacted_dict[key] = self._redact_string(value)
                    else:
                        redacted_dict[key] = value # Preserve non-string values
                return redacted_dict
            else:
                logger.warning(
                    f"[{self.node_name}] Unsupported data type for PII redaction: {type(data)}. "
                    "Returning data unchanged. Supported types: str, list[str], dict[str, str]."
                )
                return data
        except Exception as e:
            logger.error(f"[{self.node_name}] An error occurred during PII redaction: {e}", exc_info=True)
            # Depending on policy, might re-raise, but for PII redaction, returning original
            # data might be safer than crashing the pipeline if redaction fails for an item.
            return data

