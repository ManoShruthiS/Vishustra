import re
import logging
from typing import Any, Dict, Union, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node responsible for redacting Personally Identifiable Information (PII)
    from various data structures (strings, dictionaries, lists).

    The node uses configurable regular expressions to identify and replace PII with a specified
    placeholder string. It traverses nested data structures to ensure comprehensive redaction.
    """

    _NODE_NAME = "PII Redactor"

    def __init__(self, pii_patterns: Dict[str, str] = None, redaction_placeholder: str = "[REDACTED]") -> None:
        """
        Initializes the PII Redactor node.

        Args:
            pii_patterns: A dictionary where keys are PII types (e.g., 'email', 'phone')
                          and values are their corresponding regex patterns.
                          If None, a default set of common patterns will be used.
            redaction_placeholder: The string to replace identified PII with.
        """
        self._redaction_placeholder = redaction_placeholder
        self._pii_patterns = pii_patterns if pii_patterns is not None else self._get_default_pii_patterns()
        self._compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE)
            for key, pattern in self._pii_patterns.items()
        }
        logger.debug(f"Initialized {self.node_name} with PII patterns: {list(self._pii_patterns.keys())}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return self._NODE_NAME

    def _get_default_pii_patterns(self) -> Dict[str, str]:
        """
        Provides a default set of robust regex patterns for common PII types.
        These patterns are designed to be broad for demonstration.
        """
        return {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_us": r'\b(?:\+\d{1,2}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "ssn_us": r'\b\d{3}-\d{2}-\d{4}\b', # Specific US SSN format
        }

    def _redact_text(self, text: str) -> str:
        """
        Applies all configured PII patterns to a given text string, redacting any matches.

        Args:
            text: The string content to be scanned and redacted.

        Returns:
            The string with all identified PII replaced by the placeholder.
        """
        redacted_text = text
        for pii_type, pattern in self._compiled_patterns.items():
            matches = list(pattern.finditer(redacted_text))
            if matches:
                logger.debug(f"Found {len(matches)} {pii_type} instances in text for redaction.")
                redacted_text = pattern.sub(self._redaction_placeholder, redacted_text)
        return redacted_text

    def _traverse_and_redact(self, data: Any) -> Any:
        """
        Recursively traverses data structures (dict, list, string) to find and redact PII.

        Args:
            data: The data structure or primitive to be processed.

        Returns:
            A new data structure with PII redacted, or the redacted primitive.
        """
        if isinstance(data, str):
            return self._redact_text(data)
        elif isinstance(data, dict):
            return {k: self._traverse_and_redact(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._traverse_and_redact(item) for item in data]
        else:
            # For types not explicitly handled (e.g., int, float, bool, None),
            # return them as is as they are unlikely to contain PII in this context.
            logger.debug(f"Skipping redaction for unsupported data type: {type(data)}.")
            return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII based on configured patterns.

        This node can handle various input data types:
        - If `data` is a `str`, it will scan the string for PII patterns.
        - If `data` is a `dict` or a `list`, it will recursively traverse
          its contents and redact PII found within string values.
        - Other scalar data types (e.g., int, float) will be returned unchanged.

        Args:
            data: The input data potentially containing PII.
            context: A dictionary of contextual information, not directly used for redaction
                     in this implementation but available for node orchestration.

        Returns:
            The processed data with all identified PII redacted.

        Raises:
            ValueError: If the input data is `None`, which is considered an invalid state
                        for redaction processing.
            Exception: Captures and re-raises any unexpected errors during the redaction process,
                       ensuring robust error propagation.
        """
        if data is None:
            logger.error(f"[{self.node_name}] Input data for process method cannot be None.")
            raise ValueError("Input data cannot be None for PII redaction.")

        logger.info(f"[{self.node_name}] Starting PII redaction for data of type: {type(data)}.")
        try:
            redacted_data = self._traverse_and_redact(data)
            logger.info(f"[{self.node_name}] PII redaction completed successfully.")
            return redacted_data
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during PII redaction.")
            raise # Re-raise the original exception after logging