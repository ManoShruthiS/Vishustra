import logging
import re
from typing import Any, Dict, List, Union

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra node that redacts Personally Identifiable Information (PII)
    from various data structures (strings, dictionaries, lists).

    It uses configurable regular expressions to identify and replace common PII patterns
    such as email addresses, phone numbers, and credit card numbers.
    """

    def __init__(self, patterns: Dict[str, str] = None):
        """
        Initializes the PII Redactor Node with custom or default redaction patterns.

        Args:
            patterns (Dict[str, str], optional): A dictionary where keys are PII types
                                                (e.g., 'EMAIL') and values are their
                                                corresponding regex patterns.
                                                If None, a set of common patterns is used.
        """
        self._default_redaction_patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE_NUMBER": r'\b(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4})\b',
            "CREDIT_CARD": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})\b',
            "SOCIAL_SECURITY_NUMBER": r'\b\d{3}-\d{2}-\d{4}\b',
        }
        
        self._redaction_patterns = self._default_redaction_patterns.copy()
        if patterns:
            # Allow overriding or extending default patterns with user-provided ones
            self._redaction_patterns.update(patterns)
        
        # Compile patterns for efficiency during repeated processing
        self._compiled_patterns = {
            pii_type: re.compile(pattern)
            for pii_type, pattern in self._redaction_patterns.items()
        }
        logger.debug(
            f"Initialized {self.node_name} with {len(self._compiled_patterns)} compiled patterns: "
            f"{list(self._redaction_patterns.keys())}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "Vishustra.Nodes.PIIRedactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured redaction patterns to a given string.
        """
        redacted_text = text
        for pii_type, pattern in self._compiled_patterns.items():
            # Use subn to get the count of replacements for logging purposes
            new_text, count = pattern.subn(f'[REDACTED_{pii_type}]', redacted_text)
            if count > 0:
                logger.info(f"[{self.node_name}] Redacted {count} instance(s) of '{pii_type}' in a string.")
            redacted_text = new_text
        return redacted_text

    def _redact_recursive(self, data_item: Union[str, Dict[str, Any], List[Any], Any]) -> Union[str, Dict[str, Any], List[Any], Any]:
        """
        Recursively traverses data structures (strings, dictionaries, lists)
        to identify and redact PII.
        """
        if isinstance(data_item, str):
            return self._redact_string(data_item)
        elif isinstance(data_item, dict):
            redacted_dict = {}
            for key, value in data_item.items():
                redacted_dict[key] = self._redact_recursive(value)
            return redacted_dict
        elif isinstance(data_item, list):
            redacted_list = []
            for item in data_item:
                redacted_list.append(self._redact_recursive(item))
            return redacted_list
        else:
            # For any other data type (int, bool, None, custom objects),
            # PII redaction is not applicable directly. Return as is.
            logger.debug(
                f"[{self.node_name}] Skipping PII redaction for unsupported "
                f"nested data type '{type(data_item)}'. Returning as is."
            )
            return data_item

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        The `data` can be a string, a dictionary, or a list, and the redaction
        will be applied recursively to all string values found within.
        Non-string elements in collections, or top-level non-string/dict/list
        inputs, will be passed through untouched.

        Args:
            data (Any): The input data to be processed. Expected to be a string,
                        dictionary, or list for effective redaction.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing node. This node does not
                                       directly use context for redaction.

        Returns:
            Any: The data with PII redacted, preserving the original structure.
                 Returns original data untouched if the top-level type is not
                 a string, dict, or list.

        Raises:
            Exception: Re-raises any exceptions encountered during the redaction
                       process after logging them.
        """
        logger.info(f"[{self.node_name}] Starting PII redaction process.")
        logger.debug(f"[{self.node_name}] Incoming data type: {type(data)}")

        if not isinstance(data, (str, dict, list)):
            logger.warning(
                f"[{self.node_name}] Received unsupported top-level data type '{type(data)}'. "
                "PII Redactor expects str, dict, or list for effective processing. "
                "Returning data untouched."
            )
            return data

        try:
            redacted_data = self._redact_recursive(data)
            logger.info(f"[{self.node_name}] PII redaction process completed successfully.")
            return redacted_data
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during PII redaction: {e}")
            raise # Re-raise the exception after logging for upstream orchestration handling.