import re
import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node designed to redact Personally Identifiable Information (PII)
    from various data structures.

    This node leverages regular expressions to identify common PII patterns and replaces
    them with a configurable redaction string. It supports processing of strings,
    dictionaries, and lists, recursively applying redaction to string values within
    nested structures.
    """

    def __init__(self,
                 patterns: Union[List[str], None] = None,
                 redaction_string: str = "[REDACTED_PII]") -> None:
        """
        Initializes the PIIRedactorNode.

        Args:
            patterns: An optional list of regular expression strings. If provided,
                      these patterns will be used to identify PII. If None, a set
                      of default patterns covering common PII types (email, phone,
                      SSN-like, credit card-like) will be utilized.
            redaction_string: The string used to replace identified PII.
        
        Raises:
            ValueError: If any provided custom regex pattern is invalid.
        """
        self._redaction_string = redaction_string
        self._compiled_patterns: List[re.Pattern] = []

        # Default robust patterns if none are provided
        if patterns is None:
            default_patterns = [
                # Email addresses
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                # Common US Phone numbers (e.g., XXX-XXX-XXXX, (XXX) XXX-XXXX, XXX XXX XXXX)
                r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b',
                # Social Security Numbers (XXX-XX-XXXX)
                r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b',
                # Basic credit card numbers (15-16 digits, with common prefixes for Visa, MC, Amex, Discover)
                r'\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[- ]?(?:\d{4}[- ]?){3}\b',
                # Basic IP addresses (IPv4)
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                # URLs/Web addresses (simple)
                r'(?:https?://|www\.)[a-zA-Z0-9./?=_-]*\.[a-zA-Z]{2,6}(?:/[a-zA-Z0-9./?=_-]*)?\b'
            ]
            patterns = default_patterns

        for pattern_str in patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern_str, re.IGNORECASE))
            except re.error as e:
                logger.exception(f"Failed to compile PII regex pattern '{pattern_str}': {e}")
                raise ValueError(f"Invalid regex pattern provided: '{pattern_str}'") from e
        
        logger.debug(f"PIIRedactorNode initialized with {len(self._compiled_patterns)} patterns.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PIIRedactorNode"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured PII redaction patterns to a given string.
        """
        original_text = text
        for pattern in self._compiled_patterns:
            text = pattern.sub(self._redaction_string, text)
        if original_text != text:
            logger.debug(f"Redacted PII in a string. Original length: {len(original_text)}, Redacted length: {len(text)}.")
        return text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        This method intelligently handles various data types:
        - If `data` is a string, it applies PII redaction directly.
        - If `data` is a dictionary, it recursively processes all string values.
        - If `data` is a list, it recursively processes all items.
        - For any other data type, the data is returned as-is, with a debug log.

        Args:
            data: The input data payload, which can be a string, dictionary, or list.
            context: A dictionary containing workflow-wide or node-specific context
                     information. (This node currently does not utilize the context).

        Returns:
            The processed data with identified PII redacted. The data structure
            remains consistent with the input.
        """
        try:
            if isinstance(data, str):
                return self._redact_string(data)
            elif isinstance(data, dict):
                redacted_dict = {}
                for key, value in data.items():
                    redacted_dict[key] = self.process(value, context)  # Recursive call
                return redacted_dict
            elif isinstance(data, list):
                redacted_list = [self.process(item, context) for item in data]  # Recursive call
                return redacted_list
            else:
                logger.debug(
                    f"Data of type '{type(data).__name__}' is not a string, dict, or list. "
                    "Skipping PII redaction for this element and returning as is."
                )
                return data
        except Exception as e:
            logger.error(f"An unexpected error occurred during PII redaction: {e}", exc_info=True)
            # Depending on policy, might re-raise, return original data, or a redacted error placeholder.
            # For now, returning original data as a fallback to prevent pipeline failure.
            return data