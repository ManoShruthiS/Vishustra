import re
import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts Personally Identifiable Information (PII)
    from text data. It can process strings, dictionaries, and lists, recursively
    applying redaction patterns.

    Configurable with custom PII patterns and replacement string.
    """

    # Default PII patterns using regular expressions.
    # These patterns are illustrative and may need refinement for production use
    # depending on specific data formats and false positive/negative tolerances.
    DEFAULT_PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone_number": r'\b(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}|\d{7,15})\b', # Handles various international and local formats
        "social_security_number": r'\b\d{3}-\d{2}-\d{4}\b', # US SSN format
        "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
        "credit_card": r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[- ]?(?:\d{4}[- ]?){3}\d{3,4}\b|\b(?:\d[ -]*?){13,16}\b', # More specific for known prefixes, then generic
        "date_of_birth": r'\b(?:(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12][0-9]|3[01])[-/.](?:19|20)\d{2}|(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12][0-9]|3[01]))\b', # YYYY-MM-DD or MM-DD-YYYY
        "name": r'\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\b' # Very basic name detection, prone to false positives (e.g., "New York"). For robust name detection, consider Named Entity Recognition (NER).
    }

    DEFAULT_REPLACEMENT_CHAR = "[REDACTED]"

    def __init__(
        self,
        pii_patterns: Dict[str, str] = None,
        replacement_string: str = None
    ):
        """
        Initializes the PIIRedactorNode with custom PII patterns and/or
        a custom replacement string.

        Args:
            pii_patterns (Dict[str, str], optional): A dictionary where keys are PII
                types (e.g., "email") and values are regular expression strings
                to match PII. If None, default patterns are used.
            replacement_string (str, optional): The string to replace identified PII.
                If None, defaults to "[REDACTED]".
        """
        # Compile regex patterns for efficiency
        patterns_to_use = pii_patterns if pii_patterns is not None else self.DEFAULT_PII_PATTERNS
        self._pii_patterns = {
            name: re.compile(pattern, re.IGNORECASE) # re.IGNORECASE for case-insensitive matching
            for name, pattern in patterns_to_use.items()
        }
        self._replacement_string = replacement_string if replacement_string is not None else self.DEFAULT_REPLACEMENT_CHAR
        logger.debug(
            f"PIIRedactorNode initialized with {len(self._pii_patterns)} patterns "
            f"and replacement string: '{self._replacement_string}'"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII Redactor Node"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured PII patterns to redact PII from a given string.
        """
        redacted_text = text
        for pii_type, pattern_compiled in self._pii_patterns.items():
            try:
                # Only log if actual redaction occurs
                if pattern_compiled.search(redacted_text):
                    logger.debug(f"Attempting to redact '{pii_type}' patterns from text.")
                    redacted_text = pattern_compiled.sub(self._replacement_string, redacted_text)
            except re.error as e:
                logger.error(f"Regex compilation or execution error for pattern '{pii_type}': {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error during redaction for pattern '{pii_type}': {e}", exc_info=True)
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, redacting PII found within strings.
        Recursively processes dictionaries and lists.

        Args:
            data (Any): The input data to be processed. Can be a string,
                        dictionary, list, or any other type.
            context (Dict[str, Any]): A dictionary containing contextual information
                                    for the processing. Not directly used for
                                    redaction patterns in this node, but passed along
                                    for potential future enhancements or downstream nodes.

        Returns:
            Any: The processed data with PII redacted. The data type
                remains the same as the input. Returns the original data
                if an unrecoverable error occurs during processing.
        """
        if data is None:
            logger.debug("Received None data, returning None.")
            return None

        try:
            if isinstance(data, str):
                return self._redact_string(data)
            elif isinstance(data, dict):
                logger.debug("Processing dictionary data for PII redaction.")
                redacted_dict = {}
                for key, value in data.items():
                    # Recursively process values
                    redacted_dict[key] = self.process(value, context)
                return redacted_dict
            elif isinstance(data, list):
                logger.debug("Processing list data for PII redaction.")
                redacted_list = []
                for item in data:
                    # Recursively process list items
                    redacted_list.append(self.process(item, context))
                return redacted_list
            else:
                # For any other data type (int, float, bool, custom objects), return as is.
                logger.debug(f"Data of type '{type(data).__name__}' passed through without redaction as it's not a string, dict, or list.")
                return data
        except Exception as e:
            logger.critical(
                f"Fatal error encountered during PII redaction process. "
                f"Returning original data to prevent loss. Error: {e}",
                exc_info=True
            )
            # In case of a critical error, return the original data
            # to ensure data flow is not entirely broken, though it means PII might not be redacted.
            return data