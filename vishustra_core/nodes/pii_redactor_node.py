import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts personally identifiable information (PII)
    from string data within the input.

    This node uses configurable regular expressions to detect common PII patterns
    such as email addresses, phone numbers, social security numbers (US format),
    credit card numbers, and IP addresses. It can process single strings,
    dictionaries, or lists containing strings, and recursively redacts
    string values within nested structures.

    Sensitive data is replaced with a descriptive placeholder, e.g., '[REDACTED_EMAIL]'.
    """

    _DEFAULT_PII_PATTERNS: Dict[str, str] = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b',
        "ssn_us": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
        "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    }

    def __init__(self,
                 patterns: Dict[str, str] = None,
                 placeholder_prefix: str = "[REDACTED_",
                 placeholder_suffix: str = "]") -> None:
        """
        Initializes the PII Redactor Node with specified or default PII patterns
        and placeholder customization.

        Args:
            patterns (Dict[str, str], optional): A dictionary mapping PII types (e.g., "email")
                to their corresponding regular expression patterns. If None, a set of
                default patterns will be used. The keys are used to form the placeholder.
            placeholder_prefix (str, optional): The prefix string for the redaction placeholder.
                Defaults to "[REDACTED_".
            placeholder_suffix (str, optional): The suffix string for the redaction placeholder.
                Defaults to "]".
        """
        self._patterns = {
            name: re.compile(pattern)
            for name, pattern in (patterns if patterns is not None else self._DEFAULT_PII_PATTERNS).items()
        }
        self._placeholder_prefix = placeholder_prefix
        self._placeholder_suffix = placeholder_suffix
        logger.debug(f"[{self.node_name}] Initialized with {len(self._patterns)} PII patterns.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies PII redaction to a single string using configured patterns.

        Args:
            text (str): The string to redact.

        Returns:
            str: The string with identified PII replaced by placeholders.
        """
        redacted_text = text
        for pii_type, pattern_compiled in self._patterns.items():
            placeholder = f"{self._placeholder_prefix}{pii_type.upper()}{self._placeholder_suffix}"
            # Only log if a replacement actually occurs for this pattern
            if pattern_compiled.search(redacted_text):
                logger.debug(f"[{self.node_name}] Redacting '{pii_type}' patterns.")
                redacted_text = pattern_compiled.sub(placeholder, redacted_text)
        return redacted_text

    def _recursive_redact(self, data: Any) -> Any:
        """
        Recursively traverses data structures (strings, lists, dictionaries)
        to apply PII redaction to string values.

        Args:
            data (Any): The data structure to traverse and redact.

        Returns:
            Any: The data structure with PII redacted in string values.
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            return {k: self._recursive_redact(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._recursive_redact(item) for item in data]
        else:
            # For other immutable types (int, float, bool, None), return as is.
            # For mutable non-string types (e.g., custom objects), they are also
            # returned as is, as redaction applies specifically to string content.
            return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        The method handles various data types:
        - If `data` is a string, PII is redacted directly.
        - If `data` is a dictionary or a list, it traverses recursively
          and redacts PII within string values.
        - Other data types (e.g., int, float, boolean, None) are returned unchanged.

        Args:
            data (Any): The input data. This can be a string, dict, list,
                        or any other Python type.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This
                                       implementation does not use `context` for
                                       redaction patterns, but it is available
                                       for future extensions (e.g., dynamic pattern loading).

        Returns:
            Any: The data with PII redacted. The structure of the data
                 remains the same, but string values may be altered.
        """
        logger.info(f"[{self.node_name}] Starting PII redaction for data of type: {type(data)}")

        if data is None:
            logger.debug(f"[{self.node_name}] Input data is None, returning as is.")
            return None

        try:
            redacted_data = self._recursive_redact(data)
            logger.info(f"[{self.node_name}] PII redaction completed.")
            return redacted_data
        except Exception as e:
            logger.exception(f"[{self.node_name}] An error occurred during PII redaction: {e}")
            # Depending on error handling policy, might re-raise or return original data.
            # For a critical transformation like redaction, it might be safer to fail
            # or return unredacted with a clear error if the integrity of the process
            # is compromised. Here, we re-raise after logging.
            raise