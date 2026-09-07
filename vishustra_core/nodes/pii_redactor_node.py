import logging
import re
from typing import Any, Dict

# Assuming this path is where BaseNode resides in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts personally identifiable information (PII)
    from input data based on configurable regex patterns.
    It can traverse common data structures like dictionaries and lists to find PII in strings.
    """

    def __init__(self,
                 redaction_patterns: Dict[str, str] = None,
                 redaction_mask: str = "[REDACTED]",
                 enable_detailed_logging: bool = False):
        """
        Initializes the PIIRedactorNode with specific redaction patterns and a mask.

        Args:
            redaction_patterns (Dict[str, str], optional): A dictionary where keys are
                PII types (e.g., "email", "phone") and values are regex strings to
                identify that PII. If not provided, defaults to common patterns.
            redaction_mask (str, optional): The string used to replace identified PII.
                Defaults to "[REDACTED]".
            enable_detailed_logging (bool, optional): If True, enables more verbose
                logging, including details about found PII instances. Defaults to False.
        """
        self._redaction_mask = redaction_mask
        self._enable_detailed_logging = enable_detailed_logging

        # Default common PII patterns
        _default_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            # Consider adding more robust patterns for SSN, credit cards, etc.,
            # depending on regional and data-specific requirements.
            # Example: "ssn": r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",
            # Example: "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})\b"
        }
        self._redaction_patterns = _default_patterns
        if redaction_patterns:
            self._redaction_patterns.update(redaction_patterns)

        self._compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self._redaction_patterns.items()
        }
        logger.info(f"{self.node_name} initialized with patterns for: {list(self._redaction_patterns.keys())}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII_Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured redaction patterns to a given string.
        """
        redacted_text = text
        total_redacted_count = 0
        for pii_type, pattern in self._compiled_patterns.items():
            # Use subn to get the new string and count of replacements
            new_text, count = pattern.subn(self._redaction_mask, redacted_text)
            if count > 0:
                if self._enable_detailed_logging:
                    logger.debug(f"Redacted {count} instance(s) of '{pii_type}' PII.")
                redacted_text = new_text
                total_redacted_count += count
        
        if total_redacted_count > 0:
            logger.info(f"Redacted {total_redacted_count} PII instances from a string.")
        
        return redacted_text

    def _traverse_and_redact(self, data: Any) -> Any:
        """
        Recursively traverses data structures (dicts, lists) and redacts strings.
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            return {k: self._traverse_and_redact(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._traverse_and_redact(item) for item in data]
        else:
            # For other immutable types (int, float, bool, None), return as is.
            # For custom objects, we might need a specific strategy, but for PII
            # usually it's text-based.
            return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by identifying and redacting PII.
        The method supports strings, and recursively traverses dictionaries and lists
        to find and redact PII within string values.

        Args:
            data (Any): The input data to be processed. Can be a string,
                        dictionary, or list containing strings, or any other type.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used by
                                       this specific node, but required by BaseNode.

        Returns:
            Any: The data with identified PII redacted. If input data is None,
                 None is returned after logging a warning.
        
        Raises:
            Exception: Propagates any unexpected errors encountered during redaction.
        """
        if data is None:
            logger.warning(f"{self.node_name}: Received None as input data. Returning None.")
            return None

        try:
            processed_data = self._traverse_and_redact(data)
            logger.debug(f"{self.node_name} completed redaction for data type: {type(data)}")
            return processed_data
        except Exception as e:
            logger.error(f"{self.node_name}: Error during PII redaction: {e}", exc_info=True)
            # Re-raise the exception to indicate a critical failure in the processing pipeline.
            raise