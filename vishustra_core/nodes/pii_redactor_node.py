import re
import logging
from typing import Any, Dict, List, Union

# Assuming vishustra_core is accessible as a package for node imports
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra node that redacts personally identifiable information (PII)
    from input data. It identifies common PII patterns like emails, phone numbers,
    and social security numbers and replaces them with a configurable placeholder.

    This node recursively processes strings within dictionaries and lists.
    """

    DEFAULT_REDACTION_PLACEHOLDER = "[REDACTED_PII]"

    # Pre-compiled regex patterns for common PII types.
    # These are illustrative and can be extended or configured via __init__.
    _default_pii_patterns: List[re.Pattern] = [
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email addresses
        re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # Phone numbers (various formats)
        re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),  # US Social Security Numbers (simplified XXX-XX-XXXX)
        # Add more patterns as needed for specific project requirements, e.g.:
        # re.compile(r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[- ]?(?:\d{4}[- ]?){3}\b"), # Simplified credit card numbers
        # re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b") # IPv4 addresses
    ]

    def __init__(self, custom_regex_patterns: List[str] = None, redaction_placeholder: str = None):
        """
        Initializes the PII Redactor node.

        Args:
            custom_regex_patterns (List[str], optional): A list of additional regex patterns (as strings)
                                                         to use for PII detection. These will be compiled.
                                                         Invalid patterns will be logged and ignored.
            redaction_placeholder (str, optional): The string to replace identified PII with.
                                                   Defaults to '[REDACTED_PII]'.
        """
        self._current_patterns = list(self._default_pii_patterns) # Start with default patterns
        
        if custom_regex_patterns:
            for pattern_str in custom_regex_patterns:
                try:
                    self._current_patterns.append(re.compile(pattern_str))
                except re.error as e:
                    logger.warning(f"Failed to compile custom regex pattern '{pattern_str}'. It will be ignored: {e}")
        
        self._redaction_placeholder = redaction_placeholder if redaction_placeholder is not None else self.DEFAULT_REDACTION_PLACEHOLDER
        
        logger.debug(
            f"PIIRedactorNode initialized with {len(self._current_patterns)} patterns "
            f"and placeholder '{self._redaction_placeholder}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor"

    def _redact_string(self, text: str, placeholder: str) -> str:
        """
        Applies all configured PII patterns to redact sensitive information from a string.
        """
        if not isinstance(text, str):
            logger.debug(f"Attempted to redact non-string data of type {type(text).__name__}. Returning as-is.")
            return text
            
        redacted_text = text
        for pattern in self._current_patterns:
            redacted_text = pattern.sub(placeholder, redacted_text)
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        The method handles strings, dictionaries, and lists recursively.
        Other data types are returned as-is. The redaction placeholder
        can be temporarily overridden via the `context` dictionary.

        Args:
            data (Any): The input data to be processed. Can be a string, dict, list,
                        or any other data type.
            context (Dict[str, Any]): A dictionary containing context-specific information.
                                      If `context` contains a key 'redaction_placeholder'
                                      (str), its value will override the node's configured
                                      placeholder for this specific `process` call.

        Returns:
            Any: The data with identified PII redacted. Data types not explicitly
                 handled (e.g., numbers, booleans) are returned without modification.
        """
        logger.info(f"Initiating PII redaction for data of type: {type(data).__name__}.")
        
        # Allow context to temporarily override the placeholder for this specific processing run
        current_placeholder = context.get('redaction_placeholder', self._redaction_placeholder)

        try:
            if isinstance(data, str):
                return self._redact_string(data, current_placeholder)
            elif isinstance(data, dict):
                redacted_dict = {}
                for key, value in data.items():
                    # Recursively process dictionary values
                    redacted_dict[key] = self.process(value, context)
                return redacted_dict
            elif isinstance(data, list):
                # Recursively process list elements
                redacted_list = [self.process(item, context) for item in data]
                return redacted_list
            else:
                logger.debug(
                    f"Data type {type(data).__name__} is not a string, dict, or list; "
                    "skipping deep PII redaction and returning data as-is."
                )
                return data
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during PII redaction for data of type {type(data).__name__}: {e}",
                exc_info=True
            )
            # In case of an unexpected error, return the original data to prevent data loss,
            # but ensure the error is logged for investigation.
            return data