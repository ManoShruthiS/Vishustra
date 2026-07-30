import re
import logging
from typing import Any, Dict, List, Union

# This import path is specified by the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra node designed to identify and redact Personally Identifiable Information (PII)
    from various data structures using configurable regular expressions.

    It supports redaction for strings, lists of strings, and dictionary values that are strings.
    """

    def __init__(self, custom_patterns: Dict[str, str] = None):
        """
        Initializes the PII Redactor Node with a set of default PII detection patterns
        and allows for optional custom patterns to be added or overridden.

        Args:
            custom_patterns (Dict[str, str], optional): A dictionary where keys are
                descriptive tags (e.g., "SSN", "DATE_OF_BIRTH") and values are
                regular expression strings for detecting specific PII types.
                These patterns will augment or override the default patterns.
                Defaults to None.
        """
        # Default patterns for common PII types. These are designed to be broadly
        # applicable but may need refinement for specific use cases or locales.
        self._patterns = {
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "PHONE_NUMBER": r"\+?\d{1,4}[-.\s]?\(?\d{2,3}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}",
            "IP_ADDRESS": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            # Social Security Number (US-specific example)
            "SSN_US": r"\b(?!\d{3}-\d{2}-0000|\d{9})\d{3}-\d{2}-\d{4}\b",
            # Basic date patterns (e.g., YYYY-MM-DD, MM/DD/YYYY) - highly contextual
            "DATE_GENERIC": r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b",
        }

        if custom_patterns:
            self._patterns.update(custom_patterns)
            logger.debug(f"Custom PII patterns added/overridden: {list(custom_patterns.keys())}")

        # Compile all regex patterns for efficient repeated use during processing.
        self._compiled_patterns: Dict[str, re.Pattern] = {
            tag: re.compile(pattern) for tag, pattern in self._patterns.items()
        }
        logger.info(f"PII Redactor Node initialized with patterns for: {list(self._patterns.keys())}")

    @property
    def node_name(self) -> str:
        """
        Returns the human-readable name of this processing node.
        """
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies redaction to a single string using all configured PII patterns.
        Identified PII will be replaced with a tag indicating the redacted type
        (e.g., '[REDACTED_EMAIL]').

        Args:
            text (str): The input string to be redacted.

        Returns:
            str: The redacted string.
        """
        redacted_text = text
        for tag, pattern_re in self._compiled_patterns.items():
            # Only perform replacement if a match is found to avoid unnecessary string operations.
            if pattern_re.search(redacted_text):
                original_len = len(redacted_text)
                redacted_text = pattern_re.sub(f"[REDACTED_{tag}]", redacted_text)
                if len(redacted_text) < original_len:
                    logger.debug(f"Redacted '{tag}' instances in string.")
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact Personally Identifiable Information (PII).

        This method intelligently handles various input data types:
        -   `str`: The entire string is scanned and PII is redacted.
        -   `list`: Each element of the list is checked. If an element is a string,
                    it is redacted. Other types are passed through unchanged.
        -   `dict`: Only values directly associated with top-level keys are checked.
                    If a value is a string, it is redacted. Nested structures or
                    non-string values are passed through without modification.
        -   Other types: The data is returned as is, with a warning logged.

        Args:
            data (Any): The input data to be processed for PII redaction.
                        Expected types include `str`, `list[str]`, or `dict[str, str]`.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This node
                                       does not currently utilize the context.

        Returns:
            Any: The processed data with identified PII redacted. The return type
                 will match the input type if supported.

        Raises:
            Exception: Propagates any unexpected errors that occur during the
                       redaction process after logging them for debugging.
        """
        logger.info(f"PII Redactor node initiating processing for data of type: {type(data).__name__}")

        if data is None:
            logger.debug("Input data is None. No redaction performed, returning None.")
            return None

        try:
            if isinstance(data, str):
                logger.debug("Processing data as a string for PII redaction.")
                return self._redact_string(data)

            elif isinstance(data, list):
                logger.debug("Processing data as a list for PII redaction.")
                redacted_list = []
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        redacted_list.append(self._redact_string(item))
                    else:
                        logger.debug(
                            f"List item at index {i} is not a string (type: {type(item).__name__}). "
                            "Passing through without redaction."
                        )
                        redacted_list.append(item)
                return redacted_list

            elif isinstance(data, dict):
                logger.debug("Processing data as a dictionary for PII redaction (top-level string values only).")
                redacted_dict = {}
                for key, value in data.items():
                    if isinstance(value, str):
                        redacted_dict[key] = self._redact_string(value)
                    else:
                        logger.debug(
                            f"Dictionary value for key '{key}' is not a string (type: {type(value).__name__}). "
                            "Passing through without redaction."
                        )
                        redacted_dict[key] = value
                return redacted_dict

            else:
                logger.warning(
                    f"Unsupported data type '{type(data).__name__}' for PII redaction. "
                    "This node only redacts strings, lists of strings, or dictionaries "
                    "with string values. Returning data without modification."
                )
                return data

        except Exception as e:
            # Log the exception with traceback for detailed error analysis
            logger.exception(f"An unexpected error occurred during PII redaction in {self.node_name} node.")
            # Re-raise the exception to allow upstream orchestrators or error handlers to manage it.
            raise