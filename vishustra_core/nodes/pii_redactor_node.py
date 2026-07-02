import logging
import re
from typing import Any, Dict, List, Union

# Assuming BaseNode is available in the specified path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PII_RedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts Personally Identifiable Information (PII)
    from input data.

    This node uses configurable regular expressions to identify and replace common
    PII patterns such as email addresses and phone numbers. It can operate on
    strings, lists of strings, or dictionaries with string values.
    """

    _DEFAULT_EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    _DEFAULT_PHONE_PATTERN = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b' # Basic North American style phone

    def __init__(self, patterns: Union[List[str], None] = None, replacement_string: str = '[REDACTED]'):
        """
        Initializes the PII_RedactorNode with specified patterns and a replacement string.

        Args:
            patterns: A list of regex patterns (as strings) to use for PII detection.
                      If None, default patterns for email and phone numbers will be used.
                      Patterns should be valid regular expressions.
            replacement_string: The string to replace identified PII with.
        """
        self._replacement_string = replacement_string
        if patterns:
            try:
                self._compiled_patterns = [re.compile(p) for p in patterns]
            except re.error as e:
                logger.error(f"Failed to compile one or more regex patterns: {e}")
                # Fallback to defaults or raise, depending on desired strictness.
                # For robustness, we'll fall back to defaults.
                logger.warning("Falling back to default PII redaction patterns due to compilation error.")
                self._compiled_patterns = [
                    re.compile(self._DEFAULT_EMAIL_PATTERN),
                    re.compile(self._DEFAULT_PHONE_PATTERN)
                ]
        else:
            self._compiled_patterns = [
                re.compile(self._DEFAULT_EMAIL_PATTERN),
                re.compile(self._DEFAULT_PHONE_PATTERN)
            ]
        logger.info(f"PII_RedactorNode initialized with {len(self._compiled_patterns)} patterns.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII Redactor Node"

    def _redact_string(self, text: str) -> str:
        """
        Applies redaction to a single string using all compiled patterns.
        """
        redacted_text = text
        for pattern in self._compiled_patterns:
            redacted_text = pattern.sub(self._replacement_string, redacted_text)
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        The `data` can be:
        - A string: PII will be redacted directly.
        - A list: Each string item in the list will be redacted. Non-string items are preserved.
        - A dictionary: Values that are strings will be redacted. Non-string values are preserved.
        - Any other type: The data will be returned as is, with a warning logged.

        Args:
            data: The input data to be processed.
            context: A dictionary containing contextual information. Not directly used
                     for redaction logic in this node, but part of the standard API.

        Returns:
            The data with PII redacted, or the original data if it's not a supported type.
        """
        if isinstance(data, str):
            logger.debug("Redacting PII from a string.")
            return self._redact_string(data)
        elif isinstance(data, list):
            logger.debug("Redacting PII from a list of items.")
            # Create a new list to avoid modifying the input list in place
            return [self._redact_string(item) if isinstance(item, str) else item for item in data]
        elif isinstance(data, dict):
            logger.debug("Redacting PII from dictionary values.")
            # Create a new dictionary to avoid modifying the input dictionary in place
            redacted_data = {}
            for key, value in data.items():
                redacted_data[key] = self._redact_string(value) if isinstance(value, str) else value
            return redacted_data
        else:
            logger.warning(
                f"PII Redactor Node received unsupported data type: {type(data).__name__}. "
                "Returning data without redaction."
            )
            return data